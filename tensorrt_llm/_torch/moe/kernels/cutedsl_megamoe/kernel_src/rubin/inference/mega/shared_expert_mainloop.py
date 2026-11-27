"""Shared-expert members common to the prefill and gen-phase FC12 mainloops (one copy of the overlaid-ring rules)."""

import dataclasses
from typing import Tuple

import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cutlass_dsl import Boolean, Int32

from .....helpers.cute_py_helpers import Tcgen05MmaInstruction, tcgen05_smem_alloc_type
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.utils import ceil_div, strides_equal_ignoring_singletons
from .....quant_def import QuantKind
from ....schedulers.fc12_mapping import BlockPhase


class SharedKindMainloopMixin:
    """The shared kind's MMA setup, the stride rules of the AB ring both kinds overlay, and per-kind operand views."""

    def _parse_shared_problem(self, problem_desc) -> None:
        """The shared expert's dims and kind; a different kind gets its own tile K, MMA and SF layouts."""
        self.shared_hidden_size = problem_desc.get("shared_hidden_size")
        self.shared_intermediate_gateup_size = problem_desc.get("shared_intermediate_gateup_size")
        self.shared_output_size = problem_desc.get("shared_output_size")
        shared_fields = (self.shared_hidden_size, self.shared_intermediate_gateup_size, self.shared_output_size)
        if any(field is not None for field in shared_fields) and not all(field is not None for field in shared_fields):
            raise ValueError("shared_hidden_size, shared_intermediate_gateup_size and shared_output_size go together.")
        self.shared_enabled = all(field is not None for field in shared_fields)
        shared_quant_kind = problem_desc.get("shared_quant_kind")
        if shared_quant_kind is not None and not self.shared_enabled:
            raise ValueError("shared_quant_kind requires the shared expert dimensions.")
        self.shared_quant_kind = QuantKind(shared_quant_kind) if shared_quant_kind is not None else self.quant_kind
        self.heterogeneous_shared = self.shared_enabled and self.shared_quant_kind != self.quant_kind

    def _configure_shared_kind(self) -> None:
        """The shared kind's instruction, tile K and SMEM types; the shared MMA keeps the routed M, N and K steps."""
        if not self.shared_enabled:
            return
        kind = self.shared_quant_kind
        self.shared_a_dtype = kind.weight_dtype
        self.shared_b_dtype = kind.activation_dtype
        self.shared_sf_dtype = kind.sf_dtype
        self.shared_sf_vec_size = kind.sf_vec_size
        self.shared_instruction_k = kind.instruction_k(self.mma_k_mode)
        # The stage plan may lower this so a shared stage fits inside a routed one.
        self.shared_instructions_per_ab_stage = self.num_mma_instructions_per_ab_stage
        self.shared_mma_tile_k = self.shared_instructions_per_ab_stage * self.shared_instruction_k
        self.shared_mma_instruction = Tcgen05MmaInstruction(
            a_type=self.shared_a_dtype,
            b_type=self.shared_b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=self.shared_mma_instruction_mnk,
            participates=self.mma_cta_count,
            sfa_type=self.shared_sf_dtype,
            sfb_type=self.shared_sf_dtype,
            sf_vec_size=self.shared_sf_vec_size,
        )
        self.shared_a_smem_alloc_dtype = tcgen05_smem_alloc_type(
            self.shared_a_dtype, self.shared_b_dtype, self.architecture
        )
        self.shared_b_smem_alloc_dtype = tcgen05_smem_alloc_type(
            self.shared_b_dtype, self.shared_a_dtype, self.architecture
        )
        self._check_shared_tile_rows()

    def _check_shared_tile_rows(self) -> None:
        """Kernel-specific limits on the shared tile's rows; none by default."""

    @property
    def shared_mma_instruction_mnk(self) -> Tuple[int, int, int]:
        return (*self.mma_instruction_mnk[:2], self.shared_instruction_k)

    @property
    def shared_mma_tiler_mnk(self) -> Tuple[int, int, int]:
        return (*self.mma_tiler_mnk[:2], self.shared_mma_tile_k)

    @property
    def shared_sfb_instruction_shape_mnk(self) -> Tuple[int, int, int]:
        return (*self.sfb_instruction_shape_mnk[:2], self.shared_instruction_k)

    @property
    def shared_mma_tiler_sfb(self) -> Tuple[int, int, int]:
        return (*self.mma_tiler_sfb[:2], self.shared_mma_tile_k)

    def _set_shared_k_tile_counts(self) -> None:
        """K tiles of the shared FC1 (over K_s) and FC2 (over I_s); zero without a shared expert."""
        if self.shared_enabled:
            self.shared_fc1_k_tile_count = ceil_div(self.shared_hidden_size, self.shared_mma_tile_k)
            self.shared_fc2_k_tile_count = ceil_div(self.shared_intermediate_gateup_size // 2, self.shared_mma_tile_k)
        else:
            self.shared_fc1_k_tile_count = 0
            self.shared_fc2_k_tile_count = 0

    @staticmethod
    def _align_stage_stride(routed_region, shared_region):
        """Give the narrower region the wider one's stage stride so both kinds step through one ring alike."""
        routed_bits = routed_region.stride[-1] * int(routed_region.dtype.width)
        shared_bits = shared_region.stride[-1] * int(shared_region.dtype.width)
        if routed_bits < shared_bits:
            stride = shared_bits // int(routed_region.dtype.width)
            routed_region = dataclasses.replace(routed_region, stride=(*routed_region.stride[:-1], stride))
        elif shared_bits < routed_bits:
            stride = routed_bits // int(shared_region.dtype.width)
            shared_region = dataclasses.replace(shared_region, stride=(*shared_region.stride[:-1], stride))
        return routed_region, shared_region

    @staticmethod
    def _check_overlay_stage_stride(operand_name, routed_region, shared_region) -> None:
        """Both kinds must step the overlaid ring alike: equal stage bytes, a whole swizzle period for each kind."""
        stage_bytes = tuple(
            region.stride[-1] * int(region.dtype.width) // 8 for region in (routed_region, shared_region)
        )
        if stage_bytes[0] != stage_bytes[1]:
            raise RuntimeError(f"{operand_name} overlay stage bytes differ after alignment: {stage_bytes}.")
        for kind_name, region in (("routed", routed_region), ("shared", shared_region)):
            period = (1 << sum(region.swizzle)) if region.swizzle else region.byte_alignment
            if stage_bytes[0] % period != 0:
                raise RuntimeError(
                    f"{operand_name} overlay stage of {stage_bytes[0]} bytes is not a multiple of the {kind_name} "
                    f"kind's {period}-byte swizzle period."
                )

    @staticmethod
    def _align_layout_stage_stride(routed_layout, routed_dtype, shared_layout, shared_dtype):
        """``_align_stage_stride`` for the materialized CuTe layouts (plain or swizzle-composed): restride the stage
        mode."""

        def outer(layout):
            return layout.outer if isinstance(layout, cute.ComposedLayout) else layout

        routed_bits = outer(routed_layout).stride[-1] * int(routed_dtype.width)
        shared_bits = outer(shared_layout).stride[-1] * int(shared_dtype.width)

        def restride(layout, stride):
            plain = outer(layout)
            stage_layout = cute.slice_(plain, (None, None, None, 0))
            plain = cute.append(stage_layout, cute.make_layout(plain.shape[-1], stride=stride))
            if isinstance(layout, cute.ComposedLayout):
                return cute.make_composed_layout(layout.inner, layout.offset, plain)
            return plain

        if routed_bits < shared_bits:
            routed_layout = restride(routed_layout, shared_bits // int(routed_dtype.width))
        elif shared_bits < routed_bits:
            shared_layout = restride(shared_layout, routed_bits // int(shared_dtype.width))
        return routed_layout, shared_layout

    @staticmethod
    def _check_smem_plan(entries) -> None:
        """The materialized CuTe layouts must match the planned SMEM regions (an internal invariant)."""
        for tensor_name, materialized_layout, planned_region in entries:
            if materialized_layout.shape != planned_region.shape or not strides_equal_ignoring_singletons(
                planned_region.shape, materialized_layout.stride, planned_region.stride
            ):
                raise RuntimeError(
                    f"{tensor_name} SMEM plan mismatch: Python "
                    f"{planned_region.shape}:{planned_region.stride}, CuTe "
                    f"{materialized_layout.shape}:{materialized_layout.stride}."
                )

    def _register_ab_regions(self, smem_workspace: SmemWorkspace) -> None:
        """The four AB operand regions; under a heterogeneous shared expert each is two overlay lifetimes of one ring."""

        def register(target, region_name, region) -> None:
            target.register_tensor(
                region_name,
                region.dtype,
                region.shape,
                stride=region.stride,
                swizzle=region.swizzle,
                byte_alignment=region.byte_alignment,
            )

        for region_name, region, shared_region_name, shared_region in (
            (
                self.a_smem_tensor_region,
                self.a_smem_region,
                self.shared_a_smem_tensor_region,
                self.shared_a_smem_region,
            ),
            (
                self.b_smem_tensor_region,
                self.b_smem_region,
                self.shared_b_smem_tensor_region,
                self.shared_b_smem_region,
            ),
            (
                self.sfa_smem_tensor_region,
                self.sfa_smem_region,
                self.shared_sfa_smem_tensor_region,
                self.shared_sfa_smem_region,
            ),
            (
                self.sfb_smem_tensor_region,
                self.sfb_smem_region,
                self.shared_sfb_smem_tensor_region,
                self.shared_sfb_smem_region,
            ),
        ):
            if shared_region is None:
                register(smem_workspace, region_name, region)
                continue
            overlay = smem_workspace.create_overlay(region_name + ".overlay")
            register(overlay.add_lifetime("routed"), region_name, region)
            register(overlay.add_lifetime("shared"), shared_region_name, shared_region)

    @cute.jit
    def _is_shared_phase(self, phase: Int32) -> Boolean:
        return (phase == Int32(BlockPhase.SharedLinear1)) | (phase == Int32(BlockPhase.SharedLinear2))

    @cute.jit
    def _mma_operand_views(self, kind: tuple, accumulator_pointer: cute.Pointer):
        """Fragments, TMEM scale-factor windows and S2T copies of one kind (both kinds share the TMEM base)."""
        (
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            sf_dtype,
            a_smem_tensor,
            b_smem_tensor,
            sfa_smem_tensor,
            sfb_smem_tensor,
        ) = kind
        fragment_a = tiled_mma.make_fragment_A(a_smem_tensor)
        fragment_b = tiled_mma.make_fragment_B(b_smem_tensor)
        sfa_stage_smem_layout = cute.slice_(
            blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1), (None, None, None, 0)
        )
        sfb_stage_smem_layout = cute.slice_(
            blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1), (None, None, None, 0)
        )
        sfa_pointer = cute.recast_ptr(accumulator_pointer + self.num_accumulator_tmem_cols, dtype=sf_dtype)
        sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size, sfa_stage_smem_layout
        )
        tmem_sfa = cute.make_tensor(sfa_pointer, sfa_layout)
        sfb_pointer = cute.recast_ptr(
            accumulator_pointer + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols, dtype=sf_dtype
        )
        sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size, sfb_stage_smem_layout
        )
        tmem_sfb = cute.make_tensor(sfb_pointer, sfb_layout)
        tiled_copy_sfa, partitioned_smem_sfa, partitioned_tmem_sfa = self._s2t_copy_and_partition(
            sfa_smem_tensor, tmem_sfa, sf_dtype
        )
        tiled_copy_sfb, partitioned_smem_sfb, partitioned_tmem_sfb = self._s2t_copy_and_partition(
            sfb_smem_tensor, tmem_sfb, sf_dtype
        )
        return (
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
        )


__all__ = ["SharedKindMainloopMixin"]
