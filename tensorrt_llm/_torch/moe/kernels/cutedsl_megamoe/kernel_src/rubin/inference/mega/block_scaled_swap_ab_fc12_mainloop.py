"""Mainloop component for the Rubin block-scaled swap-AB FC12 kernel."""

from typing import ClassVar, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import Boolean, Int32

from .....api import ImplDesc, KernelComponent, OptionalRequirement, ProblemDesc, StaticOrRuntimeIntegerType
from .....helpers.ptx_helpers import barrier_sync_unaligned, red_add_relaxed_gpu_s32
from .....helpers.cute_py_helpers import (
    Tcgen05MmaInstruction,
    make_smem_layouts,
    tcgen05_block_scaled_acc_dtype,
    tcgen05_smem_alloc_type,
)
from .....helpers.dsl_helpers import spin_wait, tma_multicast_mask
from .....helpers.iket_compat import iket
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.utils import ceil_div, round_up
from .....quant_def import QuantKind
from ....schedulers.base import SchedulerConsumer
from ....schedulers.fc12_mapping import BlockPhase
from ....blackwell.custom_mix_cga_helpers import (
    PipelineTmaUmmaMixedCga,
    TmaAtomOrPair,
    bind_executable_tma_load_fields,
    make_executable_tma_atom_or_pair,
    select_tma_atom_for_cluster,
)
from . import dynamic_mainloop
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .shared_expert_mainloop import SharedKindMainloopMixin


class BlockScaledSwapAbFc12Mainloop(SharedKindMainloopMixin, KernelComponent):
    """Own all Rubin load, MMA, pipeline, SMEM, and TMEM mainloop state."""

    ab_pipeline_mbarriers_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.ab_pipeline_mbarriers"
    a_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.a_smem_tensor"
    b_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.b_smem_tensor"
    sfa_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.sfa_smem_tensor"
    sfb_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.sfb_smem_tensor"
    # Heterogeneous shared expert: the shared kind's views of the same operand stages.
    shared_a_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.shared_a_smem_tensor"
    shared_b_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.shared_b_smem_tensor"
    shared_sfa_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.shared_sfa_smem_tensor"
    shared_sfb_smem_tensor_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.shared_sfb_smem_tensor"
    acc_pipeline_mbarriers_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.acc_pipeline_mbarriers"
    tmem_holding_buffer_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.tmem_holding_buffer"
    tmem_deallocation_mbarrier_region: ClassVar[str] = "rubin.swap_ab_fc12.mainloop.tmem_deallocation_mbarrier"
    tmem_allocation_barrier_id: ClassVar[int] = 2

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "quant_kind": str,
            "a_major_mode": OperandMajorMode,
            "b_major_mode": OperandMajorMode,
            "hidden_size": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            # Shared expert (optional): FC1 K, FC1 N (gate-up), FC2 N and quant kind (mxfp8_e4m3 or mxfp8_e5m2).
            "shared_hidden_size": OptionalRequirement(Optional[int]),
            "shared_intermediate_gateup_size": OptionalRequirement(Optional[int]),
            "shared_output_size": OptionalRequirement(Optional[int]),
            "shared_quant_kind": OptionalRequirement(Optional[str]),
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "mma_instruction_mnk": tuple,
            "mma_tiler_mnk": tuple,
            "mma_k_mode": str,
            "cluster_shape_mn": tuple,
            "fallback_cluster_shape_mn": OptionalRequirement(Optional[tuple]),
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
        self._parse_shared_problem(problem_desc)
        self.mma_instruction_mnk = impl_desc["mma_instruction_mnk"]
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.mma_k_mode = impl_desc["mma_k_mode"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.fallback_cluster_shape_mn = impl_desc.get("fallback_cluster_shape_mn")
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
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
        tmem_plan = impl_desc["tmem_plan"]
        self.num_sfa_tmem_cols = tmem_plan.sfa_columns
        self.num_sfb_tmem_cols = tmem_plan.sfb_columns
        self.num_sf_tmem_cols = tmem_plan.sfa_columns + tmem_plan.sfb_columns
        self.num_tmem_alloc_cols = tmem_plan.allocation_columns
        self.num_accumulator_stages = tmem_plan.accumulator_stage_count
        self.num_accumulator_pipeline_stages = tmem_plan.accumulator_pipeline_stages
        if tmem_plan.accumulator_stage_stride_columns != tmem_plan.accumulator_stage_columns:
            raise ValueError("Rubin MegaMoE does not support overlapping accumulator stages.")
        if tmem_plan.accumulator_pipeline_stages != tmem_plan.accumulator_stage_count:
            raise ValueError("Rubin MegaMoE requires one pipeline stage per disjoint accumulator stage.")
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
        self.resolved_fallback_cluster_shape_mn = (
            self.cluster_shape_mn if self.fallback_cluster_shape_mn is None else self.fallback_cluster_shape_mn
        )
        self.fallback_cluster_layout_shape_vmnk = (
            (self.mma_cta_count,),
            self.resolved_fallback_cluster_shape_mn[0] // self.mma_cta_count,
            self.resolved_fallback_cluster_shape_mn[1],
            1,
        )
        self.fallback_cluster_layout_sfb_shape_vmnk = ((1,), *self.resolved_fallback_cluster_shape_mn, 1)
        self.is_mixed_cga = self.resolved_fallback_cluster_shape_mn != self.cluster_shape_mn
        self.num_mcast_ctas_a = self.cluster_layout_shape_vmnk[2]
        self.num_mcast_ctas_b = self.cluster_layout_shape_vmnk[1]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self._configure_shared_kind()
        self._select_ab_stage_plan()
        self.num_tma_load_bytes = self.ab_stage_tma_bytes * self.mma_cta_count
        if self.heterogeneous_shared:
            # Shared tiles override each producer's static expectation (``expected_tx``) with their own bytes.
            self.shared_weight_tx_count = self.shared_a_stage_tma_bytes + self.shared_sfa_stage_tma_bytes
            self.shared_token_tx_count = self.shared_b_stage_tma_bytes + self.shared_sfb_stage_tma_bytes
            if self.shared_weight_tx_count + self.shared_token_tx_count != (
                self.shared_ab_stage_tma_bytes * self.mma_cta_count
            ):
                raise ValueError("Shared-expert AB stage transaction byte accounting is inconsistent.")

    def _validate_configuration(self) -> None:
        if self.mma_k_mode != "2x":
            raise ValueError(f"Rubin MegaMoE only supports mma_k_mode='2x', got {self.mma_k_mode!r}.")
        for field_name, dimensions, expected_rank in (
            ("mma_instruction_mnk", self.mma_instruction_mnk, 3),
            ("mma_tiler_mnk", self.mma_tiler_mnk, 3),
            ("cluster_shape_mn", self.cluster_shape_mn, 2),
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
            raise NotImplementedError("Rubin MegaMoE does not implement M/N instruction repetition or B-reuse.")
        if self.mma_tiler_mnk[1] not in (64, 128, 256):
            raise ValueError("Rubin MegaMoE supports tile N in (64, 128, 256).")
        if self.mma_tiler_mnk[2] % self.instruction_k != 0:
            raise ValueError("mma_tiler K must be divisible by instruction K.")
        if self.cluster_shape_mn[0] % self.mma_cta_count != 0:
            raise ValueError("cluster M must be divisible by the MMA CTA count.")
        if self.fallback_cluster_shape_mn is not None:
            if len(self.fallback_cluster_shape_mn) != 2:
                raise ValueError("fallback_cluster_shape_mn must contain two dimensions.")
            if not all(
                isinstance(dimension, int) and not isinstance(dimension, bool)
                for dimension in self.fallback_cluster_shape_mn
            ):
                raise TypeError("fallback_cluster_shape_mn dimensions must be Python integers.")
            if any(dimension <= 0 for dimension in self.fallback_cluster_shape_mn):
                raise ValueError("fallback_cluster_shape_mn dimensions must be positive.")
            if self.fallback_cluster_shape_mn[0] % self.mma_cta_count != 0:
                raise ValueError("fallback cluster M must be divisible by the MMA CTA count.")
            if any(
                preferred_dimension % fallback_dimension != 0
                for preferred_dimension, fallback_dimension in zip(
                    self.cluster_shape_mn, self.fallback_cluster_shape_mn
                )
            ):
                raise ValueError("Preferred cluster dimensions must be divisible by fallback dimensions.")
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
        smem_workspace.register_mbarrier(self.ab_pipeline_mbarriers_region, self.num_ab_pipeline_stages * 2)
        # Register the 4 B holding word before the 128 B-aligned stage tensors so it lands in the alignment gap.
        smem_workspace.register_tensor(self.tmem_holding_buffer_region, cutlass.Int32, (1,), byte_alignment=8)

        # Heterogeneous shared expert: routed and shared regions are two views (overlay lifetimes) of one ring.
        self._register_ab_regions(smem_workspace)
        smem_workspace.register_mbarrier(self.acc_pipeline_mbarriers_region, self.num_accumulator_pipeline_stages * 2)
        smem_workspace.register_mbarrier(self.tmem_deallocation_mbarrier_region, 1)

    def _fit_shared_stage(self, routed_regions) -> tuple:
        """The most shared MMA instructions per stage with whole SF chunks in tile K that fit the routed slots."""
        full_count = self.num_mma_instructions_per_ab_stage
        sf_chunk_k = self.shared_sf_vec_size * 4
        for count in range(full_count, 0, -1):
            if full_count % count != 0 or (count * self.shared_instruction_k) % sf_chunk_k != 0:
                continue
            regions = self._shared_stage_regions(count)
            if all(shared.nbytes <= routed.nbytes for routed, shared in zip(routed_regions, regions)):
                return regions
        return self._shared_stage_regions(full_count)

    def _shared_stage_regions(self, count: int) -> tuple:
        """Set the shared instructions per AB stage and return one shared stage's SMEM regions."""
        self.shared_instructions_per_ab_stage = count
        self.shared_mma_tile_k = count * self.shared_instruction_k
        return make_smem_layouts(
            self.shared_mma_instruction,
            self.shared_mma_tiler_mnk,
            1,
            (self.a_major_mode, self.b_major_mode),
            self.architecture,
        )

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
        self.ab_stage_tma_bytes = (
            packed_bytes(stage_one_regions[0], self.a_dtype)
            + packed_bytes(stage_one_regions[1], self.b_dtype)
            + stage_one_regions[2].nbytes
            + stage_one_regions[3].nbytes
        )
        if self.heterogeneous_shared:
            shared_stage_one_regions = self._fit_shared_stage(stage_one_regions)
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
            # One ring serves both kinds: every stage is as wide as the larger kind's, each kind keeps its own swizzle.
            self.ab_stage_payload_bytes = sum(
                max(routed.nbytes, shared.nbytes) for routed, shared in zip(stage_one_regions, shared_stage_one_regions)
            )
        mbarrier_bytes = int(cutlass.Int64.width) // 8
        ab_stage_cost_bytes = self.ab_stage_payload_bytes + 2 * mbarrier_bytes
        # Accumulator + TMEM-deallocation mbarriers plus the 8 B-aligned TMEM holding word.
        tmem_holding_buffer_bytes = 8
        plan_tail_bytes = (2 * self.num_accumulator_pipeline_stages + 2) * mbarrier_bytes + tmem_holding_buffer_bytes
        self.num_ab_pipeline_stages = (self.mainloop_smem_budget_bytes - plan_tail_bytes) // ab_stage_cost_bytes
        if self.num_ab_pipeline_stages < 1:
            raise ValueError(
                f"One AB stage needs {ab_stage_cost_bytes + plan_tail_bytes} bytes, exceeding the "
                f"{self.mainloop_smem_budget_bytes}-byte mainloop budget."
            )
        (self.a_smem_region, self.b_smem_region, self.sfa_smem_region, self.sfb_smem_region) = make_smem_layouts(
            self.mma_instruction,
            self.mma_tiler_mnk,
            self.num_ab_pipeline_stages,
            (self.a_major_mode, self.b_major_mode),
            self.architecture,
        )
        self.shared_a_smem_region = None
        self.shared_b_smem_region = None
        self.shared_sfa_smem_region = None
        self.shared_sfb_smem_region = None
        if self.heterogeneous_shared:
            (
                self.shared_a_smem_region,
                self.shared_b_smem_region,
                self.shared_sfa_smem_region,
                self.shared_sfb_smem_region,
            ) = make_smem_layouts(
                self.shared_mma_instruction,
                self.shared_mma_tiler_mnk,
                self.num_ab_pipeline_stages,
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
        self.fallback_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.resolved_fallback_cluster_shape_mn, 1)), (self.tiled_mma.thr_id.shape,)
        )
        self.fallback_cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.resolved_fallback_cluster_shape_mn, 1)), (self.tiled_mma_sfb.thr_id.shape,)
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
        if self.fallback_cluster_layout_vmnk.shape != self.fallback_cluster_layout_shape_vmnk:
            raise ValueError(
                f"Fallback cluster layout mismatch: Python {self.fallback_cluster_layout_shape_vmnk}, "
                f"CuTe {self.fallback_cluster_layout_vmnk.shape}."
            )
        if self.fallback_cluster_layout_sfb_vmnk.shape != self.fallback_cluster_layout_sfb_shape_vmnk:
            raise ValueError(
                f"Fallback SFB cluster layout mismatch: Python {self.fallback_cluster_layout_sfb_shape_vmnk}, "
                f"CuTe {self.fallback_cluster_layout_sfb_vmnk.shape}."
            )

        self.a_smem_composed_layout_staged = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.mma_tiler_mnk, self.a_smem_alloc_dtype, self.num_ab_pipeline_stages
        )
        self.b_smem_composed_layout_staged = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.mma_tiler_mnk, self.b_smem_alloc_dtype, self.num_ab_pipeline_stages
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            self.tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, self.num_ab_pipeline_stages
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma, self.mma_tiler_mnk, self.sf_vec_size, self.num_ab_pipeline_stages
        )

        # With equal kinds every shared member aliases its routed counterpart.
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
                self.shared_tiled_mma, self.shared_mma_tiler_mnk, self.shared_a_smem_alloc_dtype,
                self.num_ab_pipeline_stages,
            )
            self.shared_b_smem_composed_layout_staged = sm100_utils.make_smem_layout_b(
                self.shared_tiled_mma, self.shared_mma_tiler_mnk, self.shared_b_smem_alloc_dtype,
                self.num_ab_pipeline_stages,
            )
            self.shared_sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                self.shared_tiled_mma, self.shared_mma_tiler_mnk, self.shared_sf_vec_size, self.num_ab_pipeline_stages
            )
            self.shared_sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                self.shared_tiled_mma, self.shared_mma_tiler_mnk, self.shared_sf_vec_size, self.num_ab_pipeline_stages
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

        def make_tma_maybe_pair(
            make_atom,
            operation,
            gmem_tensor,
            smem_layout,
            mma_tiler,
            tiled_mma,
            preferred_cluster_layout,
            fallback_cluster_layout,
            internal_type,
        ) -> Tuple[cute.Tensor, TmaAtomOrPair]:
            preferred_atom, preferred_tensor = make_atom(
                operation,
                gmem_tensor,
                smem_layout,
                mma_tiler,
                tiled_mma,
                preferred_cluster_layout.shape,
                internal_type=internal_type,
            )
            if self.is_mixed_cga:
                fallback_atom, _ = make_atom(
                    operation,
                    gmem_tensor,
                    smem_layout,
                    mma_tiler,
                    tiled_mma,
                    fallback_cluster_layout.shape,
                    internal_type=internal_type,
                )
                return preferred_tensor, (preferred_atom, fallback_atom)
            return preferred_tensor, preferred_atom

        fc1_a_atom, fc1_a_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            a_operation,
            fc1_a,
            a_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=a_internal_type,
        )
        fc1_b_tensor, fc1_b_atom_or_pair = make_tma_maybe_pair(
            cute.nvgpu.make_tiled_tma_atom_B,
            b_operation,
            fc1_b,
            b_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk,
            b_internal_type,
        )
        fc1_sfa_atom, fc1_sfa_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_operation,
            fc1_sfa,
            sfa_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint16,
        )
        fc1_sfb_tensor, fc1_sfb_atom_or_pair = make_tma_maybe_pair(
            cute.nvgpu.make_tiled_tma_atom_B,
            sfb_operation,
            fc1_sfb,
            sfb_stage_layout,
            self.mma_tiler_sfb,
            self.tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk,
            self.fallback_cluster_layout_sfb_vmnk,
            cutlass.Uint16,
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
        fc2_b_tensor, fc2_b_atom_or_pair = make_tma_maybe_pair(
            cute.nvgpu.make_tiled_tma_atom_B,
            b_operation,
            fc2_b,
            b_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk,
            self.fallback_cluster_layout_vmnk,
            b_internal_type,
        )
        fc2_sfa_atom, fc2_sfa_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_operation,
            fc2_sfa,
            sfa_stage_layout,
            self.mma_tiler_mnk,
            self.tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Uint16,
        )
        fc2_sfb_tensor, fc2_sfb_atom_or_pair = make_tma_maybe_pair(
            cute.nvgpu.make_tiled_tma_atom_B,
            sfb_operation,
            fc2_sfb,
            sfb_stage_layout,
            self.mma_tiler_sfb,
            self.tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk,
            self.fallback_cluster_layout_sfb_vmnk,
            cutlass.Uint16,
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
            fc1_b_atom_or_pair,
            fc1_sfb_tensor,
            fc1_sfb_atom_or_pair,
            fc2_b_tensor,
            fc2_b_atom_or_pair,
            fc2_sfb_tensor,
            fc2_sfb_atom_or_pair,
        )

    def _make_tma_b_maybe_pair(
        self, operation, gmem_tensor, smem_layout, mma_tiler, tiled_mma, cluster_layout, fallback_layout, internal_type
    ) -> Tuple[cute.Tensor, TmaAtomOrPair]:
        """One B-side TMA atom, or a preferred/fallback pair on a mixed-cluster launch."""
        preferred_atom, tensor = cute.nvgpu.make_tiled_tma_atom_B(
            operation, gmem_tensor, smem_layout, mma_tiler, tiled_mma, cluster_layout.shape, internal_type=internal_type
        )
        if not self.is_mixed_cga:
            return tensor, preferred_atom
        fallback_atom, _ = cute.nvgpu.make_tiled_tma_atom_B(
            operation, gmem_tensor, smem_layout, mma_tiler, tiled_mma, fallback_layout.shape, internal_type=internal_type
        )
        return tensor, (preferred_atom, fallback_atom)

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
        shared_a_smem_alloc_dtype = self.a_smem_alloc_dtype
        shared_b_smem_alloc_dtype = self.b_smem_alloc_dtype
        if self.heterogeneous_shared:
            shared_a_smem_alloc_dtype = self.shared_a_smem_alloc_dtype
            shared_b_smem_alloc_dtype = self.shared_b_smem_alloc_dtype
        shared_a_dtype = self.shared_quant_kind.weight_dtype
        shared_b_dtype = self.shared_quant_kind.activation_dtype
        a_internal_type = shared_a_smem_alloc_dtype if shared_a_smem_alloc_dtype is not shared_a_dtype else None
        b_internal_type = shared_b_smem_alloc_dtype if shared_b_smem_alloc_dtype is not shared_b_dtype else None
        mma_tiler_mnk = self.shared_mma_tiler_mnk if self.heterogeneous_shared else self.mma_tiler_mnk
        mma_tiler_sfb = self.shared_mma_tiler_sfb if self.heterogeneous_shared else self.mma_tiler_sfb

        results = []
        for operand_a, operand_sfa, operand_b, operand_sfb in (
            (shared_fc1_a, shared_fc1_sfa, shared_fc1_b, shared_fc1_sfb),
            (shared_fc2_a, shared_fc2_sfa, shared_fc2_b, shared_fc2_sfb),
        ):
            a_atom, a_tensor = cute.nvgpu.make_tiled_tma_atom_A(
                a_operation,
                operand_a,
                a_stage_layout,
                mma_tiler_mnk,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=a_internal_type,
            )
            sfa_atom, sfa_tensor = cute.nvgpu.make_tiled_tma_atom_A(
                sfa_operation,
                operand_sfa,
                sfa_stage_layout,
                mma_tiler_mnk,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Uint16,
            )
            # The token legs multicast across the cluster, so a mixed launch needs a preferred/fallback pair.
            b_tensor, b_atom = self._make_tma_b_maybe_pair(
                b_operation,
                operand_b,
                b_stage_layout,
                mma_tiler_mnk,
                tiled_mma,
                self.cluster_layout_vmnk,
                self.fallback_cluster_layout_vmnk,
                b_internal_type,
            )
            sfb_tensor, sfb_atom = self._make_tma_b_maybe_pair(
                sfb_operation,
                operand_sfb,
                sfb_stage_layout,
                mma_tiler_sfb,
                self.shared_tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk,
                self.fallback_cluster_layout_sfb_vmnk,
                cutlass.Uint16,
            )
            results.extend((a_tensor, a_atom, sfa_tensor, sfa_atom, b_tensor, b_atom, sfb_tensor, sfb_atom))
        return tuple(results)

    @cute.jit
    def assign_device_members(
        self,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        cta_coord_in_cluster: cute.Coord,
        is_fallback_cluster: Boolean,
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
        self.is_fallback_cluster = is_fallback_cluster
        self.is_preferred_cluster = is_fallback_cluster == Boolean(False)
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
        """Create the shared two-producer TMA-to-UMMA pipeline."""
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 2)
        return PipelineTmaUmmaMixedCga.create(
            barrier_storage=smem_workspace.ptr(self.ab_pipeline_mbarriers_region, smem_base),
            num_stages=self.num_ab_pipeline_stages,
            producer_group=producer_group,
            num_mma_consumer_warps=1,
            tx_count=self.num_tma_load_bytes // 2,
            preferred_cta_layout_vmnk=self.cluster_layout_vmnk,
            fallback_cta_layout_vmnk=self.fallback_cluster_layout_vmnk,
            cta_coord_vmnk=self.main_vmnk_coord,
            is_fallback_cluster=self.is_fallback_cluster,
            mcast_mode_mn=(1, 1),
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

    @cute.jit
    def _issue_tma_a_tile(
        self,
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: tuple,
        a_atom: cute.CopyAtom,
        a_tensor: cute.Tensor,
        sfa_atom: cute.CopyAtom,
        sfa_tensor: cute.Tensor,
        a_smem_tensor: cute.Tensor,
        sfa_smem_tensor: cute.Tensor,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        work_tile,
        k_tile_count,
        a_cta_layout: cute.Layout,
        multicast_mask,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        expected_tx: Optional[Int32] = None,
    ) -> pipeline.PipelineState:
        """Load one tile's weight and weight-scale k tiles of one kind (routed or shared)."""
        thread_mma = tiled_mma.get_slice(self.mma_cta_index)
        real_a, a_descriptor_ptr = kernel_extension.get_gmem_tensor("a", a_tensor, work_tile)
        real_sfa, sfa_descriptor_ptr = kernel_extension.get_gmem_tensor("sfa", sfa_tensor, work_tile)

        global_a = cute.local_tile(real_a, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None))
        global_sfa = cute.local_tile(real_sfa, cute.slice_(mma_tiler_mnk, (None, 0, None)), (None, None, None))
        partitioned_global_a = thread_mma.partition_A(global_a)
        partitioned_global_sfa = thread_mma.partition_A(global_sfa)

        partitioned_smem_a, partitioned_global_a = cpasync.tma_partition(
            a_atom,
            self.main_vmnk_coord[2],
            a_cta_layout,
            cute.group_modes(a_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_a, 0, 3),
        )
        partitioned_smem_sfa, partitioned_global_sfa = cpasync.tma_partition(
            sfa_atom,
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
            ab_pipeline.producer_acquire(ab_pipeline_state, empty_status, expected_tx=expected_tx)
            tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
            stage_index = ab_pipeline_state.index
            ab_pipeline_state.advance()
            if k_tile_idx + 1 < k_tile_count:
                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
            cute.copy(
                a_atom,
                global_a_slice[(None, k_tile_idx)],
                partitioned_smem_a[(None, stage_index)],
                tma_bar_ptr=tma_barrier,
                tma_desc_ptr=a_descriptor_ptr,
                mcast_mask=multicast_mask,
            )
            cute.copy(
                sfa_atom,
                global_sfa_slice[(None, k_tile_idx)],
                partitioned_smem_sfa[(None, stage_index)],
                tma_bar_ptr=tma_barrier,
                tma_desc_ptr=sfa_descriptor_ptr,
                mcast_mask=multicast_mask,
            )
        return ab_pipeline_state

    @cute.jit
    def _issue_routed_tma_a_tile(
        self, routed_kind: tuple, routed_args: tuple, work_tile, ab_pipeline_state: pipeline.PipelineState
    ) -> pipeline.PipelineState:
        """Routed FC1 or FC2 weight tile (opens the tile's iket range)."""
        (
            fc1_a_atom,
            fc1_a_tensor,
            fc1_sfa_atom,
            fc1_sfa_tensor,
            fc2_a_atom,
            fc2_a_tensor,
            fc2_sfa_atom,
            fc2_sfa_tensor,
            kernel_extension,
            a_cta_layout,
            multicast_mask,
            ab_pipeline,
        ) = routed_args
        if work_tile.phase == Int32(BlockPhase.Linear1):
            iket.range_push("tma_weight_fc1")
            ab_pipeline_state = self._issue_tma_a_tile(
                *routed_kind[:2],
                fc1_a_atom,
                fc1_a_tensor,
                fc1_sfa_atom,
                fc1_sfa_tensor,
                *routed_kind[2:],
                kernel_extension,
                work_tile,
                self.fc1_k_tile_count,
                a_cta_layout,
                multicast_mask,
                ab_pipeline,
                ab_pipeline_state,
            )
        else:
            iket.range_push("tma_weight_fc2")
            ab_pipeline_state = self._issue_tma_a_tile(
                *routed_kind[:2],
                fc2_a_atom,
                fc2_a_tensor,
                fc2_sfa_atom,
                fc2_sfa_tensor,
                *routed_kind[2:],
                kernel_extension,
                work_tile,
                self.fc2_k_tile_count,
                a_cta_layout,
                multicast_mask,
                ab_pipeline,
                ab_pipeline_state,
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
        shared_fc1_tma_a_tensor: Optional[cute.Tensor] = None,
        shared_fc1_tma_a_atom: Optional[cute.CopyAtom] = None,
        shared_fc1_tma_sfa_tensor: Optional[cute.Tensor] = None,
        shared_fc1_tma_sfa_atom: Optional[cute.CopyAtom] = None,
        shared_fc2_tma_a_tensor: Optional[cute.Tensor] = None,
        shared_fc2_tma_a_atom: Optional[cute.CopyAtom] = None,
        shared_fc2_tma_sfa_tensor: Optional[cute.Tensor] = None,
        shared_fc2_tma_sfa_atom: Optional[cute.CopyAtom] = None,
        shared_kernel_extension: Optional[BlockScaledSwapAbFc12Extension] = None,
    ) -> None:
        """Run the TMA-A warp that loads weights and their scale factors."""
        multicast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.use_2cta_instrs):
            multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn, None, self.cta_coord_in_cluster, None, self.use_2cta_instrs, "a"
            )

        a_cta_layout = cute.make_layout(cute.slice_(self.cluster_layout_vmnk, (0, 0, None, 0)).shape)
        routed_kind = (self.tiled_mma, self.mma_tiler_mnk, self.a_smem_tensor, self.sfa_smem_tensor)
        shared_kind = None
        shared_expected_tx = None
        if cutlass.const_expr(self.shared_enabled):
            shared_kind = (
                self.shared_tiled_mma,
                self.shared_mma_tiler_mnk,
                self.shared_a_smem_tensor,
                self.shared_sfa_smem_tensor,
            )
            if cutlass.const_expr(self.heterogeneous_shared):
                shared_expected_tx = Int32(self.shared_weight_tx_count)

        routed_args = (
            fc1_tma_a_atom,
            fc1_tma_a_tensor,
            fc1_tma_sfa_atom,
            fc1_tma_sfa_tensor,
            fc2_tma_a_atom,
            fc2_tma_a_tensor,
            fc2_tma_sfa_atom,
            fc2_tma_sfa_tensor,
            kernel_extension,
            a_cta_layout,
            multicast_mask,
            ab_pipeline,
        )
        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            phase = work_tile.phase
            if cutlass.const_expr(self.shared_enabled):
                if self._is_shared_phase(phase):
                    if phase == Int32(BlockPhase.SharedLinear1):
                        iket.range_push("tma_weight_shared_fc1")
                        ab_pipeline_state = self._issue_tma_a_tile(
                            *shared_kind[:2],
                            shared_fc1_tma_a_atom,
                            shared_fc1_tma_a_tensor,
                            shared_fc1_tma_sfa_atom,
                            shared_fc1_tma_sfa_tensor,
                            *shared_kind[2:],
                            shared_kernel_extension,
                            work_tile,
                            self.shared_fc1_k_tile_count,
                            a_cta_layout,
                            multicast_mask,
                            ab_pipeline,
                            ab_pipeline_state,
                            shared_expected_tx,
                        )
                    else:
                        iket.range_push("tma_weight_shared_fc2")
                        ab_pipeline_state = self._issue_tma_a_tile(
                            *shared_kind[:2],
                            shared_fc2_tma_a_atom,
                            shared_fc2_tma_a_tensor,
                            shared_fc2_tma_sfa_atom,
                            shared_fc2_tma_sfa_tensor,
                            *shared_kind[2:],
                            shared_kernel_extension,
                            work_tile,
                            self.shared_fc2_k_tile_count,
                            a_cta_layout,
                            multicast_mask,
                            ab_pipeline,
                            ab_pipeline_state,
                            shared_expected_tx,
                        )
                else:
                    ab_pipeline_state = self._issue_routed_tma_a_tile(
                        routed_kind, routed_args, work_tile, ab_pipeline_state
                    )
            else:
                ab_pipeline_state = self._issue_routed_tma_a_tile(
                    routed_kind, routed_args, work_tile, ab_pipeline_state
                )

            iket.range_pop()
            work_tile = sched_consumer.consume_work()

        ab_pipeline.producer_tail(ab_pipeline_state)

    @cute.jit
    def _issue_tma_b_tile(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        mma_tiler_mnk: tuple,
        mma_tiler_sfb: tuple,
        b_partition_atom: cute.CopyAtom,
        b_exec_atom: cute.CopyAtom,
        b_tensor: cute.Tensor,
        sfb_partition_atom: cute.CopyAtom,
        sfb_exec_atom: cute.CopyAtom,
        sfb_tensor: cute.Tensor,
        b_smem_tensor: cute.Tensor,
        sfb_smem_tensor: cute.Tensor,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        work_tile,
        k_tile_count,
        b_layouts: tuple,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        expected_tx: Optional[Int32] = None,
    ) -> pipeline.PipelineState:
        """Load one tile's token and token-scale k tiles of one kind (routed or shared)."""
        (
            b_cta_layout,
            sfb_cta_layout,
            b_partition_coord,
            sfb_partition_coord,
            b_multicast_mask,
            sfb_multicast_mask,
        ) = b_layouts
        thread_mma = tiled_mma.get_slice(self.mma_cta_index)
        thread_mma_sfb = tiled_mma_sfb.get_slice(self.mma_cta_index)
        real_b, b_descriptor_ptr = kernel_extension.get_gmem_tensor("b", b_tensor, work_tile)
        real_sfb, sfb_descriptor_ptr = kernel_extension.get_gmem_tensor("sfb", sfb_tensor, work_tile)

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
            b_partition_atom,
            b_partition_coord,
            b_cta_layout,
            cute.group_modes(b_smem_tensor, 0, 3),
            cute.group_modes(partitioned_global_b, 0, 3),
        )
        partitioned_smem_sfb, partitioned_global_sfb = cpasync.tma_partition(
            sfb_partition_atom,
            sfb_partition_coord,
            sfb_cta_layout,
            cute.group_modes(sfb_smem_tensor, 0, 3),
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
            ab_pipeline.producer_acquire(ab_pipeline_state, empty_status, expected_tx=expected_tx)
            tma_barrier = ab_pipeline.producer_get_barrier(ab_pipeline_state)
            stage_index = ab_pipeline_state.index
            ab_pipeline_state.advance()
            if k_tile_idx + 1 < k_tile_count:
                empty_status = ab_pipeline.producer_try_acquire(ab_pipeline_state)
            b_issue_atom = bind_executable_tma_load_fields(
                b_exec_atom,
                tma_bar_ptr=tma_barrier,
                tma_desc_ptr=b_descriptor_ptr,
                mcast_mask=b_multicast_mask,
            )
            cute.copy(b_issue_atom, global_b_slice[(None, k_tile_idx)], partitioned_smem_b[(None, stage_index)])
            sfb_issue_atom = bind_executable_tma_load_fields(
                sfb_exec_atom,
                tma_bar_ptr=tma_barrier,
                tma_desc_ptr=sfb_descriptor_ptr,
                mcast_mask=sfb_multicast_mask,
            )
            cute.copy(sfb_issue_atom, global_sfb_slice[(None, k_tile_idx)], partitioned_smem_sfb[(None, stage_index)])
        return ab_pipeline_state

    @cute.jit
    def _issue_routed_tma_b_tile(
        self,
        routed_kind: tuple,
        routed_smem: tuple,
        routed_args: tuple,
        work_tile,
        ab_pipeline_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Routed FC1 or FC2 token tile after its readiness wait (opens the tile's iket range)."""
        (
            fc1_b_partition_atom,
            fc1_b_exec_atom,
            fc1_b_tensor,
            fc1_sfb_partition_atom,
            fc1_sfb_exec_atom,
            fc1_sfb_tensor,
            fc2_b_partition_atom,
            fc2_b_exec_atom,
            fc2_b_tensor,
            fc2_sfb_partition_atom,
            fc2_sfb_exec_atom,
            fc2_sfb_tensor,
            kernel_extension,
            fc1_done_counter_pointer,
            fc2_spin_threshold,
            b_layouts,
            ab_pipeline,
        ) = routed_args
        if work_tile.phase == Int32(BlockPhase.Linear1):
            iket.range_push("tma_token_fc1")
            iket.range_push("tma_token_fc1_wait")
            kernel_extension.wait_for_input(work_tile)
            iket.range_pop()
            ab_pipeline_state = self._issue_tma_b_tile(
                *routed_kind,
                fc1_b_partition_atom,
                fc1_b_exec_atom,
                fc1_b_tensor,
                fc1_sfb_partition_atom,
                fc1_sfb_exec_atom,
                fc1_sfb_tensor,
                *routed_smem,
                kernel_extension,
                work_tile,
                self.fc1_k_tile_count,
                b_layouts,
                ab_pipeline,
                ab_pipeline_state,
            )
        else:
            iket.range_push("tma_token_fc2")
            counter_slot = work_tile.cumulative_token_block_count + work_tile.tile_n_idx
            iket.range_push("tma_token_fc2_wait")
            spin_wait(
                fc1_done_counter_pointer + counter_slot,
                lambda value: value >= fc2_spin_threshold,
                sleep_cycles=500,
                peek_status=work_tile.peek_ready,
            )
            iket.range_pop()
            ab_pipeline_state = self._issue_tma_b_tile(
                *routed_kind,
                fc2_b_partition_atom,
                fc2_b_exec_atom,
                fc2_b_tensor,
                fc2_sfb_partition_atom,
                fc2_sfb_exec_atom,
                fc2_sfb_tensor,
                *routed_smem,
                kernel_extension,
                work_tile,
                self.fc2_k_tile_count,
                b_layouts,
                ab_pipeline,
                ab_pipeline_state,
            )
        return ab_pipeline_state

    @cute.jit
    def run_tma_b(
        self,
        *,
        fc1_tma_b_tensor: cute.Tensor,
        fc1_tma_b_atom_or_pair: TmaAtomOrPair,
        fc1_tma_sfb_tensor: cute.Tensor,
        fc1_tma_sfb_atom_or_pair: TmaAtomOrPair,
        fc2_tma_b_tensor: cute.Tensor,
        fc2_tma_b_atom_or_pair: TmaAtomOrPair,
        fc2_tma_sfb_tensor: cute.Tensor,
        fc2_tma_sfb_atom_or_pair: TmaAtomOrPair,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        sched_consumer: SchedulerConsumer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        fc1_done_counter_pointer: cute.Pointer,
        fc2_spin_threshold: Int32,
        shared_fc1_tma_b_tensor: Optional[cute.Tensor] = None,
        shared_fc1_tma_b_atom: Optional[TmaAtomOrPair] = None,
        shared_fc1_tma_sfb_tensor: Optional[cute.Tensor] = None,
        shared_fc1_tma_sfb_atom: Optional[TmaAtomOrPair] = None,
        shared_fc2_tma_b_tensor: Optional[cute.Tensor] = None,
        shared_fc2_tma_b_atom: Optional[TmaAtomOrPair] = None,
        shared_fc2_tma_sfb_tensor: Optional[cute.Tensor] = None,
        shared_fc2_tma_sfb_atom: Optional[TmaAtomOrPair] = None,
        shared_kernel_extension: Optional[BlockScaledSwapAbFc12Extension] = None,
        shared_fc1_done_counter_pointer: Optional[cute.Pointer] = None,
        shared_fc2_spin_threshold: Optional[Int32] = None,
    ) -> None:
        """Run the TMA-B warp that loads token data and scale factors."""
        fc1_tma_b_partition_atom = fc1_tma_b_atom_or_pair
        fc1_tma_sfb_partition_atom = fc1_tma_sfb_atom_or_pair
        fc2_tma_b_partition_atom = fc2_tma_b_atom_or_pair
        fc2_tma_sfb_partition_atom = fc2_tma_sfb_atom_or_pair
        if cutlass.const_expr(isinstance(fc1_tma_b_atom_or_pair, tuple)):
            fc1_tma_b_partition_atom = fc1_tma_b_atom_or_pair[0]
            fc1_tma_sfb_partition_atom = fc1_tma_sfb_atom_or_pair[0]
            fc2_tma_b_partition_atom = fc2_tma_b_atom_or_pair[0]
            fc2_tma_sfb_partition_atom = fc2_tma_sfb_atom_or_pair[0]

        fc1_tma_b_exec_atom = select_tma_atom_for_cluster(
            make_executable_tma_atom_or_pair(fc1_tma_b_atom_or_pair), self.is_fallback_cluster
        )
        fc1_tma_sfb_exec_atom = select_tma_atom_for_cluster(
            make_executable_tma_atom_or_pair(fc1_tma_sfb_atom_or_pair), self.is_fallback_cluster
        )
        fc2_tma_b_exec_atom = select_tma_atom_for_cluster(
            make_executable_tma_atom_or_pair(fc2_tma_b_atom_or_pair), self.is_fallback_cluster
        )
        fc2_tma_sfb_exec_atom = select_tma_atom_for_cluster(
            make_executable_tma_atom_or_pair(fc2_tma_sfb_atom_or_pair), self.is_fallback_cluster
        )

        b_multicast_mask = None
        sfb_multicast_mask = None
        if cutlass.const_expr(self.is_b_mcast or self.use_2cta_instrs):
            b_multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn,
                self.resolved_fallback_cluster_shape_mn if self.is_mixed_cga else None,
                self.cta_coord_in_cluster,
                self.is_preferred_cluster if self.is_mixed_cga else None,
                self.use_2cta_instrs,
                "b",
            )
            sfb_multicast_mask = tma_multicast_mask(
                self.cluster_shape_mn,
                self.resolved_fallback_cluster_shape_mn if self.is_mixed_cga else None,
                self.cta_coord_in_cluster,
                self.is_preferred_cluster if self.is_mixed_cga else None,
                self.use_2cta_instrs,
                "sfb",
            )

        b_cta_layout = cute.make_layout(cute.slice_(self.cluster_layout_vmnk, (0, None, 0, 0)).shape)
        sfb_cta_layout = cute.make_layout(cute.slice_(self.cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        b_partition_coord = self.main_vmnk_coord[1]
        sfb_partition_coord = self.sfb_vmnk_coord[1]
        if cutlass.const_expr(self.is_mixed_cga):
            if self.is_fallback_cluster:
                split_factor = self.cluster_shape_mn[0] // self.resolved_fallback_cluster_shape_mn[0]
                b_partition_coord = b_partition_coord * Int32(split_factor)
                sfb_partition_coord = sfb_partition_coord * Int32(split_factor)
        b_layouts = (
            b_cta_layout,
            sfb_cta_layout,
            b_partition_coord,
            sfb_partition_coord,
            b_multicast_mask,
            sfb_multicast_mask,
        )
        routed_kind = (self.tiled_mma, self.tiled_mma_sfb, self.mma_tiler_mnk, self.mma_tiler_sfb)
        routed_smem = (self.b_smem_tensor, self.sfb_smem_tensor)
        shared_kind = None
        shared_smem = None
        shared_expected_tx = None
        shared_fc1_b_exec_atom = None
        shared_fc1_sfb_exec_atom = None
        shared_fc2_b_exec_atom = None
        shared_fc2_sfb_exec_atom = None
        if cutlass.const_expr(self.shared_enabled):
            shared_kind = (
                self.shared_tiled_mma,
                self.shared_tiled_mma_sfb,
                self.shared_mma_tiler_mnk,
                self.shared_mma_tiler_sfb,
            )
            shared_smem = (self.shared_b_smem_tensor, self.shared_sfb_smem_tensor)
            if cutlass.const_expr(self.heterogeneous_shared):
                shared_expected_tx = Int32(self.shared_token_tx_count)
            shared_fc1_b_exec_atom = select_tma_atom_for_cluster(
                make_executable_tma_atom_or_pair(shared_fc1_tma_b_atom), self.is_fallback_cluster
            )
            shared_fc1_sfb_exec_atom = select_tma_atom_for_cluster(
                make_executable_tma_atom_or_pair(shared_fc1_tma_sfb_atom), self.is_fallback_cluster
            )
            shared_fc2_b_exec_atom = select_tma_atom_for_cluster(
                make_executable_tma_atom_or_pair(shared_fc2_tma_b_atom), self.is_fallback_cluster
            )
            shared_fc2_sfb_exec_atom = select_tma_atom_for_cluster(
                make_executable_tma_atom_or_pair(shared_fc2_tma_sfb_atom), self.is_fallback_cluster
            )
            if cutlass.const_expr(isinstance(shared_fc1_tma_b_atom, tuple)):
                # Partitioning follows the preferred atom; the fallback one only changes the issued multicast.
                shared_fc1_tma_b_atom = shared_fc1_tma_b_atom[0]
                shared_fc1_tma_sfb_atom = shared_fc1_tma_sfb_atom[0]
                shared_fc2_tma_b_atom = shared_fc2_tma_b_atom[0]
                shared_fc2_tma_sfb_atom = shared_fc2_tma_sfb_atom[0]

        routed_args = (
            fc1_tma_b_partition_atom,
            fc1_tma_b_exec_atom,
            fc1_tma_b_tensor,
            fc1_tma_sfb_partition_atom,
            fc1_tma_sfb_exec_atom,
            fc1_tma_sfb_tensor,
            fc2_tma_b_partition_atom,
            fc2_tma_b_exec_atom,
            fc2_tma_b_tensor,
            fc2_tma_sfb_partition_atom,
            fc2_tma_sfb_exec_atom,
            fc2_tma_sfb_tensor,
            kernel_extension,
            fc1_done_counter_pointer,
            fc2_spin_threshold,
            b_layouts,
            ab_pipeline,
        )
        work_tile = sched_consumer.consume_work()
        while work_tile.is_valid_tile:
            phase = work_tile.phase
            if cutlass.const_expr(self.shared_enabled):
                if self._is_shared_phase(phase):
                    if phase == Int32(BlockPhase.SharedLinear1):
                        # Shared FC1 reads this rank's own tokens and waits for nothing.
                        iket.range_push("tma_token_shared_fc1")
                        ab_pipeline_state = self._issue_tma_b_tile(
                            *shared_kind,
                            shared_fc1_tma_b_atom,
                            shared_fc1_b_exec_atom,
                            shared_fc1_tma_b_tensor,
                            shared_fc1_tma_sfb_atom,
                            shared_fc1_sfb_exec_atom,
                            shared_fc1_tma_sfb_tensor,
                            *shared_smem,
                            shared_kernel_extension,
                            work_tile,
                            self.shared_fc1_k_tile_count,
                            b_layouts,
                            ab_pipeline,
                            ab_pipeline_state,
                            shared_expected_tx,
                        )
                    else:
                        iket.range_push("tma_token_shared_fc2")
                        iket.range_push("tma_token_shared_fc2_wait")
                        spin_wait(
                            shared_fc1_done_counter_pointer + work_tile.tile_n_idx,
                            lambda value: value >= shared_fc2_spin_threshold,
                            sleep_cycles=500,
                            peek_status=work_tile.peek_ready,
                        )
                        iket.range_pop()
                        ab_pipeline_state = self._issue_tma_b_tile(
                            *shared_kind,
                            shared_fc2_tma_b_atom,
                            shared_fc2_b_exec_atom,
                            shared_fc2_tma_b_tensor,
                            shared_fc2_tma_sfb_atom,
                            shared_fc2_sfb_exec_atom,
                            shared_fc2_tma_sfb_tensor,
                            *shared_smem,
                            shared_kernel_extension,
                            work_tile,
                            self.shared_fc2_k_tile_count,
                            b_layouts,
                            ab_pipeline,
                            ab_pipeline_state,
                            shared_expected_tx,
                        )
                else:
                    ab_pipeline_state = self._issue_routed_tma_b_tile(
                        routed_kind, routed_smem, routed_args, work_tile, ab_pipeline_state
                    )
            else:
                ab_pipeline_state = self._issue_routed_tma_b_tile(
                    routed_kind, routed_smem, routed_args, work_tile, ab_pipeline_state
                )

            iket.range_pop()
            work_tile = sched_consumer.consume_work()

        ab_pipeline.producer_tail(ab_pipeline_state)

    @cute.jit
    def _mma_tile(
        self,
        quant_kind: QuantKind,
        mma_instruction_mnk: tuple,
        sf_dtype,
        views: tuple,
        accumulator: cute.Tensor,
        accumulator_pointer: cute.Pointer,
        work_tile,
        k_tile_count,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        instructions_per_stage: int,
    ) -> pipeline.PipelineState:
        """One tile's k loop of one kind, prefetching the next stage's readiness."""
        (
            fragment_a,
            fragment_b,
            tmem_sfa,
            tmem_sfb,
            sfb_layout,
            tiled_copy_sfa,
            partitioned_smem_sfa,
            partitioned_tmem_sfa,
            tiled_copy_sfb,
            partitioned_smem_sfb,
            partitioned_tmem_sfb,
        ) = views
        mma_tmem_sfb = tmem_sfb
        if cutlass.const_expr(self.cta_tile_n == 64):
            sfb_shift = (work_tile.tile_n_idx % Int32(2)) * Int32(2)
            shifted_sfb_pointer = cute.recast_ptr(
                accumulator_pointer + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + sfb_shift,
                dtype=sf_dtype,
            )
            mma_tmem_sfb = cute.make_tensor(shifted_sfb_pointer, sfb_layout)
        use_n128_sfb_mapping = False
        if cutlass.const_expr(quant_kind == QuantKind.nvfp4 and self.cta_tile_n == 256):
            aligned_instruction_n = (work_tile.valid_tokens_in_cta_tile + Int32(15)) & Int32(-16)
            use_n128_sfb_mapping = aligned_instruction_n <= Int32(128)

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
                for instruction_index in cutlass.range_constexpr(instructions_per_stage):
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
                    sfb_window_tensor=mma_tmem_sfb,
                    valid_tokens_in_tile=work_tile.valid_tokens_in_cta_tile,
                    mma_instruction_mnk=mma_instruction_mnk,
                    window_instruction_offset=0,
                    window_instruction_count=instructions_per_stage,
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
                    sfb_window_tensor=mma_tmem_sfb,
                    valid_tokens_in_tile=work_tile.valid_tokens_in_cta_tile,
                    mma_instruction_mnk=mma_instruction_mnk,
                    window_instruction_offset=0,
                    window_instruction_count=instructions_per_stage,
                    first_instruction_accumulate=k_tile_idx != 0,
                )

                ab_pipeline.consumer_release(ab_pipeline_state)
                ab_pipeline_state.advance()
                next_ab_pipeline_state.advance()
                full_status = next_full_status
        return ab_pipeline_state

    @cute.jit
    def run_mma(
        self,
        *,
        tmem_allocator,
        ab_pipeline: pipeline.PipelineAsync,
        ab_pipeline_state: pipeline.PipelineState,
        acc_pipeline: pipeline.PipelineAsync,
        sched_consumer: SchedulerConsumer,
        ring_device_workspace=None,
        ring_retire_region: Optional[str] = None,
    ) -> None:
        """Execute SM107 dynamic-N MMA across K tiles; a heterogeneous shared expert's tiles use the shared views.

        With a ring workspace whose blocks are retired by the FC2 operand load (``ring_retire_region`` names the
        retire counter in ``ring_device_workspace``) the leader CTA also counts every routed FC2 tile inside the
        FC2 output width for its pool token block.
        """
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
        shared_views = None
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
            phase = work_tile.phase
            k_tile_count = Int32(self.fc2_k_tile_count)
            if phase == Int32(BlockPhase.Linear1):
                k_tile_count = Int32(self.fc1_k_tile_count)
            if cutlass.const_expr(self.shared_enabled):
                if phase == Int32(BlockPhase.SharedLinear1):
                    k_tile_count = Int32(self.shared_fc1_k_tile_count)
                if phase == Int32(BlockPhase.SharedLinear2):
                    k_tile_count = Int32(self.shared_fc2_k_tile_count)
            accumulator_stage_index = acc_producer_state.index

            if self.is_leader_cta:
                accumulator = accumulator_base[(None, None, accumulator_stage_index)]
                accumulator = cute.tiled_divide(accumulator, accumulator.shape)
                acc_pipeline.producer_acquire(acc_producer_state)
                if cutlass.const_expr(ring_retire_region is not None):
                    # Ring workspace: the pool token block this tile counts for after its MMAs, held by lane 0;
                    # -1 in the other lanes and for a tile that does not count (not a routed FC2 tile, or beyond
                    # the FC2 output). Taken before the K loop, and per lane on purpose: a lane-dependent value
                    # lives in a general register, which this warp barely uses, while a warp-uniform one would take
                    # a uniform register across the loop. This region is at the limit of those (later warp roles
                    # keep theirs live through it), and one more pushes the kernel's warp index to the stack.
                    ring_retire_block = Int32(-1)
                    if cute.arch.lane_idx() == Int32(0):
                        if work_tile.phase == Int32(BlockPhase.Linear2):
                            if work_tile.tile_m_idx * Int32(self.cta_tile_m) < Int32(self.hidden_size):
                                ring_retire_block = work_tile.cumulative_token_block_count + work_tile.tile_n_idx
                if cutlass.const_expr(self.heterogeneous_shared):
                    if self._is_shared_phase(phase):
                        ab_pipeline_state = self._mma_tile(
                            self.shared_quant_kind,
                            self.shared_mma_instruction_mnk,
                            self.shared_sf_dtype,
                            shared_views,
                            accumulator,
                            accumulator_pointer,
                            work_tile,
                            k_tile_count,
                            ab_pipeline,
                            ab_pipeline_state,
                            self.shared_instructions_per_ab_stage,
                        )
                    else:
                        ab_pipeline_state = self._mma_tile(
                            self.quant_kind,
                            self.mma_instruction_mnk,
                            self.sf_dtype,
                            views,
                            accumulator,
                            accumulator_pointer,
                            work_tile,
                            k_tile_count,
                            ab_pipeline,
                            ab_pipeline_state,
                            self.num_mma_instructions_per_ab_stage,
                        )
                else:
                    ab_pipeline_state = self._mma_tile(
                        self.quant_kind,
                        self.mma_instruction_mnk,
                        self.sf_dtype,
                        views,
                        accumulator,
                        accumulator_pointer,
                        work_tile,
                        k_tile_count,
                        ab_pipeline,
                        ab_pipeline_state,
                        self.num_mma_instructions_per_ab_stage,
                    )
                if cutlass.const_expr(ring_retire_region is not None):
                    # Ring workspace: count this FC2 MMA tile for its token block. The K loop waited for every
                    # operand stage of the tile, so every operand byte bound for this CTA (with two-CTA
                    # instructions: for this CTA pair) has landed and the tile no longer reads the block's
                    # FC1-output rows; a copy multicast to another pair is covered by that pair's count. When
                    # every in-bound FC2 MMA tile of the block has counted, the block is retired: nothing reads
                    # its ring slot any more, and token_in may write the next block of that slot (the slot
                    # lease). This warp counts and not the epilogue: it is idle until the epilogue frees an
                    # accumulator, the epilogue is not. It counts before the commit: the epilogue of this tile,
                    # the finalize barrier and the tail reset of the counter all follow the commit, so the count
                    # of the last tile cannot land after the reset. The add is relaxed: it makes no write of this
                    # thread visible. Lane 0 alone holds a block. The address is formed here from the buffer
                    # base (element_ptr), not from a counter pointer made at the kernel entry, which would be
                    # one more value this region has to carry.
                    if ring_retire_block >= Int32(0):
                        red_add_relaxed_gpu_s32(
                            ring_device_workspace.element_ptr(ring_retire_region, ring_retire_block), Int32(1)
                        )
                acc_pipeline.producer_commit(acc_producer_state)

            acc_producer_state.advance()
            work_tile = sched_consumer.consume_work()
        acc_pipeline.producer_tail(acc_producer_state)


__all__ = ["BlockScaledSwapAbFc12Mainloop"]

