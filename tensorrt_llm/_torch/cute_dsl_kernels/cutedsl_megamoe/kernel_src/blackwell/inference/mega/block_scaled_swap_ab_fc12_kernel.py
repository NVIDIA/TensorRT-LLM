# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lean block-scaled swap-AB FC12 kernel composition for Blackwell."""

from typing import ClassVar, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode
from cutlass.cutlass_dsl import Boolean
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .....api import (
    ImplDesc,
    KernelClass,
    OptionalRequirement,
    ProblemDesc,
    StaticOrRuntimeIntegerType,
)
from .....helpers.cute_py_helpers import (
    Tcgen05MmaInstruction,
    make_tcgen05_tmem_plan,
    tcgen05_block_scaled_acc_dtype,
)
from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.utils import ceil_div, round_up
from .....quant_def import CombineFormat, QuantKind
from ....schedulers.base import WorkIdAcquisitionMode
from ....schedulers.fc12_scheduler import BlackwellFusedFc12Scheduler
from ...custom_mix_cga_helpers import TmaAtomOrPair
from .block_scaled_swap_ab_fc12_epilogue import GatedActEpilogueArgs, SwapABGatedActEpilogue
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .block_scaled_swap_ab_fc12_mainloop import BlockScaledSwapAbFc12Mainloop


class BlockScaledSwapAbFc12Kernel(KernelClass):
    """Lean FC12-only top-level with one explicit eight-warp topology."""

    fc1_output_region: ClassVar[str] = "blackwell.swap_ab_fc12.fc1_output"
    fc1_output_sf_region: ClassVar[str] = "blackwell.swap_ab_fc12.fc1_output_sf"
    fc1_done_counter_region: ClassVar[str] = "blackwell.swap_ab_fc12.fc1_done_counter"
    epilogue_warp_ids: ClassVar[Tuple[int, int, int, int]] = (0, 1, 2, 3)
    mma_warp_id: ClassVar[int] = 4
    tma_a_warp_id: ClassVar[int] = 5
    tma_b_warp_id: ClassVar[int] = 6
    scheduler_warp_id: ClassVar[int] = 7
    threads_per_cta: ClassVar[int] = 8 * 32

    @classmethod
    def problem_desc_require(cls) -> dict[str, object]:
        return {
            "expert_count": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "hidden_size": StaticOrRuntimeIntegerType,
            "quant_kind": str,
            "a_major_mode": OperandMajorMode,
            "b_major_mode": OperandMajorMode,
            "combine_format": CombineFormat,
            "gate_up_clamp": Optional[float],
            "situ_beta": OptionalRequirement(Optional[float]),
            "situ_linear_beta": OptionalRequirement(Optional[float]),
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "group_hint": int,
            "token_padding_block": int,
            "sf_padding_block": int,
            "work_id_mode": str,
            "max_tokens": int,
            "launch_cluster_count": int,
            "fc2_use_bulk": bool,
            "fc1_epi_flag_batch": int,
            "fc2_epi_flag_batch": int,
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.expert_count = problem_desc["expert_count"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.a_dtype = self.quant_kind.weight_dtype
        self.b_dtype = self.quant_kind.activation_dtype
        self.sf_dtype = self.quant_kind.sf_dtype
        self.sf_vec_size = self.quant_kind.sf_vec_size
        self.acc_dtype = tcgen05_block_scaled_acc_dtype
        self.a_major_mode = problem_desc["a_major_mode"]
        self.b_major_mode = problem_desc["b_major_mode"]
        self.combine_format = problem_desc["combine_format"]
        self.gate_up_clamp = problem_desc["gate_up_clamp"]
        self.situ_beta = problem_desc.get("situ_beta")
        self.situ_linear_beta = problem_desc.get("situ_linear_beta")

        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.group_hint = impl_desc["group_hint"]
        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.work_id_mode: WorkIdAcquisitionMode = impl_desc["work_id_mode"]
        self.max_tokens = impl_desc["max_tokens"]
        self.launch_cluster_count = impl_desc["launch_cluster_count"]
        self.fc2_use_bulk = impl_desc["fc2_use_bulk"]
        self.fc1_epi_flag_batch = impl_desc["fc1_epi_flag_batch"]
        self.fc2_epi_flag_batch = impl_desc["fc2_epi_flag_batch"]

        self.occupancy = 1
        self.architecture = "sm_100"

        self._validate_geometry()
        mma_cta_count = 2 if self.use_2cta_instrs else 1
        mma_instruction = Tcgen05MmaInstruction(
            a_type=self.a_dtype,
            b_type=self.b_dtype,
            acc_type=self.acc_dtype,
            instruction_mnk=(
                self.mma_tiler_mnk[0],
                self.mma_tiler_mnk[1],
                self.quant_kind.instruction_k("1x"),
            ),
            participates=mma_cta_count,
            sfa_type=self.sf_dtype,
            sfb_type=self.sf_dtype,
            sf_vec_size=self.sf_vec_size,
        )
        tmem_plan = make_tcgen05_tmem_plan(mma_instruction, self.architecture, self.mma_tiler_mnk)
        resolved_impl_desc = ImplDesc(
            {
                **impl_desc,
                "hint": self.group_hint,
                "tmem_plan": tmem_plan,
                "is_swap_ab": True,
                "num_scheduler_consumer_threads": self.threads_per_cta - 32,
                "num_accumulator_consumer_warps_per_cta": len(self.epilogue_warp_ids),
                "communication_enabled": False,
            }
        )
        self.scheduler = BlackwellFusedFc12Scheduler(problem_desc, resolved_impl_desc)
        self.epilogue = SwapABGatedActEpilogue(problem_desc, resolved_impl_desc)
        if self.epilogue.fc1_output_dtype is not self.b_dtype:
            raise ValueError("Epilogue FC1 output dtype must match the Mainloop B dtype.")
        if self.epilogue.fc1_output_sf_dtype is not self.sf_dtype:
            raise ValueError("Epilogue FC1 output scale dtype must match the Mainloop scale dtype.")
        if self.epilogue.sf_vec_size != self.sf_vec_size:
            raise ValueError("Epilogue and Mainloop scale vector sizes must match.")
        self._device_workspace = self._build_device_workspace()
        self._mainloop, self._smem_workspace = self._build_mainloop_and_smem(
            problem_desc, resolved_impl_desc
        )

    def _validate_geometry(self) -> None:
        static_expert_dimensions = (
            isinstance(self.expert_count, int),
            isinstance(self.intermediate_gateup_size, int),
            isinstance(self.hidden_size, int),
        )
        if any(static_expert_dimensions) and not all(static_expert_dimensions):
            raise ValueError("FC12 expert dimensions must be either all static or all runtime.")
        if not all(static_expert_dimensions):
            raise NotImplementedError(
                "The Lean FC12 Kernel currently requires static expert dimensions."
            )
        if self.expert_count <= 0 or self.intermediate_gateup_size <= 0 or self.hidden_size <= 0:
            raise ValueError("FC12 expert dimensions must be positive.")
        if self.launch_cluster_count <= 0:
            raise ValueError("launch_cluster_count must be positive.")
        if self.intermediate_gateup_size % 2 != 0:
            raise ValueError("The SwiGLU intermediate dimension must be even.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if self.token_padding_block <= 0:
            raise ValueError("token_padding_block must be positive.")
        if self.token_padding_block % 64 != 0:
            raise ValueError("token_padding_block must be a multiple of 64.")
        if self.sf_vec_size <= 0:
            raise ValueError("sf_vec_size must be positive.")
        if self.quant_kind.needs_unpack_tma(self.architecture):
            for name, extent in (
                ("hidden_size", self.hidden_size),
                ("intermediate_downproj", self.intermediate_gateup_size // 2),
            ):
                if extent % 128 != 0:
                    raise ValueError(
                        f"{self.quant_kind} loads its fp4 weight through the unpacking TMA, which "
                        f"requires {name} to be a multiple of 128, got {extent}."
                    )
        mma_m, mma_n, mma_k = self.mma_tiler_mnk
        cluster_m, cluster_n = self.cluster_shape_mn
        expected_mma_m = 256 if self.use_2cta_instrs else 128
        if mma_m != expected_mma_m:
            raise ValueError(f"mma_tiler M must be {expected_mma_m}, got {mma_m}.")
        if mma_n not in (64, 128, 256):
            raise ValueError(f"mma_tiler N must be 64, 128, or 256, got {mma_n}.")
        if mma_k % (self.sf_vec_size * 4) != 0:
            raise ValueError("mma_tiler K must be divisible by four scale-factor vectors.")
        if self.intermediate_gateup_size % (self.sf_vec_size * 4) != 0:
            raise ValueError(
                "The intermediate dimension must be divisible by four scale-factor vectors."
            )
        if cluster_n != 1:
            raise ValueError(f"The swap-AB FC12 path requires cluster N=1, got {cluster_n}.")
        if cluster_m <= 0 or cluster_m > 16 or cluster_m & (cluster_m - 1):
            raise ValueError("cluster M must be a power of two no greater than 16.")
        if self.use_2cta_instrs and cluster_m % 2 != 0:
            raise ValueError("Two-CTA MMA requires an even cluster M.")

    def _build_device_workspace(self) -> DeviceWorkspace:
        intermediate_downproj = self.intermediate_gateup_size // 2
        sf_column_count = round_up(ceil_div(intermediate_downproj, self.sf_vec_size), 4)
        max_sf_rows = self.max_tokens + self.expert_count * self.sf_padding_block
        counter_slot_count = ceil_div(self.max_tokens, self.mma_tiler_mnk[1]) + self.expert_count

        device_workspace = DeviceWorkspace()
        device_workspace.register(
            self.fc1_output_region,
            self.epilogue.fc1_output_dtype,
            (self.max_tokens, intermediate_downproj),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        device_workspace.register(
            self.fc1_output_sf_region,
            self.epilogue.fc1_output_sf_dtype,
            (max_sf_rows, sf_column_count),
            buffer_space="local",
            mem_order=(1, 0),
            byte_alignment=128,
        )
        device_workspace.register(
            self.fc1_done_counter_region,
            cutlass.Int32,
            (counter_slot_count,),
            buffer_space="local",
            byte_alignment=16,
            reset="tail_reset",
        )
        self.scheduler.register_device_workspace(device_workspace)
        device_workspace.finalize()
        return device_workspace

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return required local and shared workspace bytes."""
        return self._device_workspace.local_and_shared_bytes

    @property
    def require_zero_workspace_leading_bytes(self) -> Tuple[int, int]:
        return self._device_workspace.require_zero_workspace_leading_bytes

    def name(self) -> str:
        return ""

    def aot_compile(self):
        return None

    def _build_mainloop_and_smem(
        self, problem_desc: ProblemDesc, resolved_impl_desc: ImplDesc
    ) -> Tuple[BlockScaledSwapAbFc12Mainloop, SmemWorkspace]:
        smem_limit = utils.get_smem_capacity_in_bytes() // self.occupancy
        smem_workspace = SmemWorkspace()
        self.scheduler.register_smem_regions(smem_workspace)
        self.epilogue.register_smem_regions(smem_workspace)
        alignment_shift_bytes = max(region.byte_alignment for region in smem_workspace.regions())
        mainloop_smem_budget_bytes = (
            smem_limit - smem_workspace.estimate_total_bytes() - alignment_shift_bytes
        )
        if mainloop_smem_budget_bytes <= 0:
            raise ValueError("Fixed kernel components leave no SMEM budget for the Mainloop.")

        mainloop_impl_desc = ImplDesc(
            {**resolved_impl_desc, "mainloop_smem_budget_bytes": mainloop_smem_budget_bytes}
        )
        mainloop = BlockScaledSwapAbFc12Mainloop(problem_desc, mainloop_impl_desc)

        mainloop.register_smem_regions(smem_workspace)
        smem_workspace.finalize(max_bytes=smem_limit)
        return mainloop, smem_workspace

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,  # (tokens, hidden)
        fc1_weight: cute.Tensor,  # (experts, hidden, intermediate_gateup)
        activation_sf: cute.Tensor,  # (padded_tokens, padded_hidden // sf_vec_size)
        fc1_weight_sf: cute.Tensor,  # (experts, packed_fc1_sf)
        fc2_weight: cute.Tensor,  # (experts, intermediate_downproj, hidden)
        fc2_weight_sf: cute.Tensor,  # (experts, packed_fc2_sf)
        fc2_output: cute.Tensor,  # (tokens, topk, hidden)
        local_workspace: cute.Pointer,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        expert_token_sizes: Optional[cute.Tensor] = None,  # (experts,)
        expert_token_prefix_sum: Optional[cute.Tensor] = None,  # (experts,)
        topk_scores: Optional[cute.Tensor] = None,  # (tokens,)
        fc1_alpha: Optional[cute.Tensor] = None,
        fc2_alpha: Optional[cute.Tensor] = None,
        fc1_norm_const: Optional[cute.Tensor] = None,
    ) -> None:
        """Prepare and launch the lean fused FC1+FC2 kernel.

        The caller must zero the workspace prefix reported by
        require_zero_workspace_leading_bytes before its first launch.
        """
        if cutlass.const_expr((expert_token_sizes is None) == (expert_token_prefix_sum is None)):
            raise ValueError(
                "Exactly one of expert_token_sizes and expert_token_prefix_sum must be provided."
            )
        # The quant kind fixes every operand format, so the tensors are where a caller mismatch can
        # still show up. See BlockScaledSwapAbMegaMoeKernel.__call__.
        for operand_name, operand, expected_dtype in (
            ("activation", activation, self.b_dtype),
            ("activation_sf", activation_sf, self.sf_dtype),
            ("fc1_weight", fc1_weight, self.a_dtype),
            ("fc1_weight_sf", fc1_weight_sf, self.sf_dtype),
            ("fc2_weight", fc2_weight, self.a_dtype),
            ("fc2_weight_sf", fc2_weight_sf, self.sf_dtype),
        ):
            if cutlass.const_expr(operand.element_type is not expected_dtype):
                raise TypeError(
                    f"{self.quant_kind} requires {operand_name} to be {expected_dtype}, got {operand.element_type}."
                )
        if cutlass.const_expr(not self.quant_kind.uses_global_scale):
            for scalar_name, scalar in (
                ("fc1_alpha", fc1_alpha),
                ("fc2_alpha", fc2_alpha),
                ("fc1_norm_const", fc1_norm_const),
            ):
                if cutlass.const_expr(scalar is not None):
                    raise ValueError(
                        f"{self.quant_kind} folds its rescale into the e8m0 scale factors, "
                        f"so {scalar_name} must be None."
                    )

        def rewrite_tensor_shape(tensor: cute.Tensor, shape: Tuple) -> cute.Tensor:
            return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=tensor.stride))

        self._device_workspace.assign_device_members(local_workspace)
        fc1_output = self._device_workspace.tensor(self.fc1_output_region)
        fc1_output_sf = self._device_workspace.tensor(self.fc1_output_sf_region)

        if cutlass.const_expr(isinstance(self.expert_count, int)):
            experts = self.expert_count
        else:
            experts = fc1_weight.shape[0]
        if cutlass.const_expr(isinstance(self.hidden_size, int)):
            hidden = self.hidden_size
        else:
            hidden = fc1_weight.shape[1]
        if cutlass.const_expr(isinstance(self.intermediate_gateup_size, int)):
            intermediate_gateup = self.intermediate_gateup_size
        else:
            intermediate_gateup = fc1_weight.shape[2]
        intermediate_downproj = intermediate_gateup // 2
        fc1_weight = rewrite_tensor_shape(fc1_weight, (experts, hidden, intermediate_gateup))
        fc2_weight = rewrite_tensor_shape(fc2_weight, (experts, intermediate_downproj, hidden))
        activation = rewrite_tensor_shape(activation, (activation.shape[0], hidden))
        if cutlass.const_expr(isinstance(activation.shape[0], int)):
            if cutlass.const_expr(activation.shape[0] > self.max_tokens):
                raise ValueError("activation rows exceed the configured max_tokens.")
        fc2_output = rewrite_tensor_shape(
            fc2_output, (fc2_output.shape[0], fc2_output.shape[1], hidden)
        )

        singleton = cutlass.Int32(1)
        token_rows = activation.shape[0]

        fc1_a = cute.make_tensor(
            fc1_weight.iterator,
            cute.make_layout(
                (cutlass.Int32(intermediate_gateup), cutlass.Int32(hidden), cutlass.Int32(experts)),
                stride=(fc1_weight.stride[2], fc1_weight.stride[1], fc1_weight.stride[0]),
            ),
        )
        fc1_b = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (token_rows, cutlass.Int32(hidden), singleton),
                stride=(activation.stride[0], activation.stride[1], 0),
            ),
        )
        fc1_output_gemm = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (token_rows, cutlass.Int32(intermediate_downproj), singleton),
                stride=(fc1_output.stride[0], fc1_output.stride[1], 0),
            ),
        )

        sf_vec_size = self.sf_vec_size
        padded_token_rows = activation_sf.shape[0]
        padded_hidden = activation_sf.shape[1] * sf_vec_size
        fc1_sfb = cute.make_tensor(
            activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (padded_token_rows, cutlass.Int32(padded_hidden), singleton), sf_vec_size
            ),
        )
        padded_intermediate_gateup = cute.round_up(intermediate_gateup, sf_vec_size * 4)
        expected_fc1_weight_sf_columns = padded_intermediate_gateup * padded_hidden // sf_vec_size
        if cutlass.const_expr(
            isinstance(fc1_weight_sf.shape[1], int)
            and isinstance(expected_fc1_weight_sf_columns, int)
        ):
            if cutlass.const_expr(fc1_weight_sf.shape[1] != expected_fc1_weight_sf_columns):
                raise ValueError("fc1_weight_sf has an incompatible column count.")
        fc1_sfa = cute.make_tensor(
            fc1_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (
                    cutlass.Int32(padded_intermediate_gateup),
                    cutlass.Int32(padded_hidden),
                    cutlass.Int32(experts),
                ),
                sf_vec_size,
            ),
        )

        experts_fc2, intermediate_fc2, hidden_fc2 = fc2_weight.shape
        fc2_a = cute.make_tensor(
            fc2_weight.iterator,
            cute.make_layout(
                (
                    cutlass.Int32(hidden_fc2),
                    cutlass.Int32(intermediate_fc2),
                    cutlass.Int32(experts_fc2),
                ),
                stride=(fc2_weight.stride[2], fc2_weight.stride[1], fc2_weight.stride[0]),
            ),
        )
        fc2_b = fc1_output_gemm
        padded_fc2_hidden = cute.round_up(hidden_fc2, 128)
        padded_fc2_intermediate = cute.round_up(intermediate_fc2, sf_vec_size * 4)
        expected_fc2_weight_sf_columns = padded_fc2_hidden * padded_fc2_intermediate // sf_vec_size
        if cutlass.const_expr(
            isinstance(fc2_weight_sf.shape[1], int)
            and isinstance(expected_fc2_weight_sf_columns, int)
        ):
            if cutlass.const_expr(fc2_weight_sf.shape[1] != expected_fc2_weight_sf_columns):
                raise ValueError("fc2_weight_sf has an incompatible column count.")
        fc2_sfa = cute.make_tensor(
            fc2_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (
                    cutlass.Int32(padded_fc2_hidden),
                    cutlass.Int32(padded_fc2_intermediate),
                    cutlass.Int32(experts_fc2),
                ),
                sf_vec_size,
            ),
        )
        fc2_sfb = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (
                    fc1_output_sf.shape[0],
                    cutlass.Int32(fc1_output_sf.shape[1] * sf_vec_size),
                    singleton,
                ),
                sf_vec_size,
            ),
        )

        self._mainloop.materialize_codegen_members()
        (
            fc1_tma_a_tensor,
            fc1_tma_a_atom,
            fc1_tma_sfa_tensor,
            fc1_tma_sfa_atom,
            fc2_tma_a_tensor,
            fc2_tma_a_atom,
            fc2_tma_sfa_tensor,
            fc2_tma_sfa_atom,
            fc1_tma_b_tensor,
            fc1_tma_b_atom_or_pair,
            fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom_or_pair,
            fc2_tma_b_tensor,
            fc2_tma_b_atom_or_pair,
            fc2_tma_sfb_tensor,
            fc2_tma_sfb_atom_or_pair,
        ) = self._mainloop.prepare_tma_load_params(
            fc1_a=fc1_a,
            fc1_b=fc1_b,
            fc1_sfa=fc1_sfa,
            fc1_sfb=fc1_sfb,
            fc2_a=fc2_a,
            fc2_b=fc2_b,
            fc2_sfa=fc2_sfa,
            fc2_sfb=fc2_sfb,
        )

        (fc1_output_tma_atom, fc1_output_tma_tensor, fc2_output_tma_atom, fc2_output_tma_tensor) = (
            self.epilogue.prepare_tma_store_params(fc1_output_gemm, fc2_output)
        )

        grid = self.scheduler.get_grid_shape(max_active_clusters=max_active_clusters)
        self._kernel(
            fc1_tma_a_tensor,
            fc1_tma_a_atom,
            fc1_tma_sfa_tensor,
            fc1_tma_sfa_atom,
            fc2_tma_a_tensor,
            fc2_tma_a_atom,
            fc2_tma_sfa_tensor,
            fc2_tma_sfa_atom,
            fc1_tma_b_tensor,
            fc1_tma_b_atom_or_pair,
            fc1_tma_sfb_tensor,
            fc1_tma_sfb_atom_or_pair,
            fc2_tma_b_tensor,
            fc2_tma_b_atom_or_pair,
            fc2_tma_sfb_tensor,
            fc2_tma_sfb_atom_or_pair,
            fc1_output_tma_atom,
            fc1_output_tma_tensor,
            fc2_output_tma_atom,
            fc2_output_tma_tensor,
            fc2_output,
            expert_token_sizes,
            expert_token_prefix_sum,
            (experts, intermediate_gateup, hidden),
            local_workspace,
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
            topk_scores,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
        )

    @cute.kernel
    def _kernel(
        self,
        fc1_tma_a_tensor: cute.Tensor,
        fc1_tma_a_atom: cute.CopyAtom,
        fc1_tma_sfa_tensor: cute.Tensor,
        fc1_tma_sfa_atom: cute.CopyAtom,
        fc2_tma_a_tensor: cute.Tensor,
        fc2_tma_a_atom: cute.CopyAtom,
        fc2_tma_sfa_tensor: cute.Tensor,
        fc2_tma_sfa_atom: cute.CopyAtom,
        fc1_tma_b_tensor: cute.Tensor,
        fc1_tma_b_atom_or_pair: TmaAtomOrPair,
        fc1_tma_sfb_tensor: cute.Tensor,
        fc1_tma_sfb_atom_or_pair: TmaAtomOrPair,
        fc2_tma_b_tensor: cute.Tensor,
        fc2_tma_b_atom_or_pair: TmaAtomOrPair,
        fc2_tma_sfb_tensor: cute.Tensor,
        fc2_tma_sfb_atom_or_pair: TmaAtomOrPair,
        fc1_output_tma_atom: cute.CopyAtom,
        fc1_output_tma_tensor: cute.Tensor,
        fc2_output_tma_atom: Optional[cute.CopyAtom],
        fc2_output_tma_tensor: Optional[cute.Tensor],
        fc2_output: cute.Tensor,
        expert_token_sizes: Optional[cute.Tensor],
        expert_token_prefix_sum: Optional[cute.Tensor],
        actual_expert_shape: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        local_workspace: cute.Pointer,
        fc1_alpha: Optional[cute.Tensor],
        fc2_alpha: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        topk_scores: Optional[cute.Tensor],
    ):
        """Compose Scheduler, Mainloop, and Epilogue."""
        self._mainloop.materialize_codegen_members()
        storage_type = self._smem_workspace.storage_class()
        smem_allocator = utils.SmemAllocator()
        storage = smem_allocator.allocate(storage_type)
        smem_base = storage.buffer.data_ptr()
        self._device_workspace.assign_device_members(local_workspace)
        fc1_done_counter = self._device_workspace.tensor(self.fc1_done_counter_region)
        fc1_output_sf_storage = self._device_workspace.tensor(self.fc1_output_sf_region)
        if cutlass.const_expr(isinstance(self.hidden_size, int)):
            hidden = self.hidden_size
        else:
            hidden = actual_expert_shape[2]
        if cutlass.const_expr(isinstance(self.intermediate_gateup_size, int)):
            intermediate_gateup = self.intermediate_gateup_size
        else:
            intermediate_gateup = actual_expert_shape[1]
        fc1_output_sf = cute.make_tensor(
            fc1_output_sf_storage.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (fc1_output_sf_storage.shape[0], intermediate_gateup // 2, cutlass.Int32(1)),
                self.epilogue.sf_vec_size,
            ),
        )

        thread_idx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(thread_idx // 32)
        block_idx = cute.arch.block_idx()

        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        cta_coord_in_cluster = (
            cta_rank_in_cluster % self.cluster_shape_mn[0],
            cta_rank_in_cluster // self.cluster_shape_mn[0],
            cutlass.Int32(0),
        )

        # Standalone FC12 binds the preferred layout before pipeline creation.
        self._mainloop.assign_device_members(
            self._smem_workspace,
            smem_base,
            cta_coord_in_cluster,
            Boolean(False),
            hidden,
            intermediate_gateup,
        )
        ab_pipeline = self._mainloop.create_ab_pipeline(self._smem_workspace, smem_base)
        tma_a_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._mainloop.num_ab_pipeline_stages
        )
        tma_b_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self._mainloop.num_ab_pipeline_stages
        )
        mma_pipeline_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self._mainloop.num_ab_pipeline_stages
        )
        acc_pipeline = self._mainloop.create_acc_pipeline(self._smem_workspace, smem_base)
        tmem_allocator = self._mainloop.create_tmem_allocator(
            self._smem_workspace, smem_base, allocator_warp_id=self.epilogue_warp_ids[0]
        )
        self.scheduler.assign_device_members(
            expert_token_sizes=expert_token_sizes,
            expert_token_prefix_sum=expert_token_prefix_sum,
            actual_expert_shape=actual_expert_shape,
            block_idx=block_idx,
            smem_workspace=self._smem_workspace,
            smem_base=smem_base,
            device_workspace=self._device_workspace,
        )
        scheduler = self.scheduler

        fc2_spin_threshold = (
            intermediate_gateup + self._mainloop.cta_tile_m - 1
        ) // self._mainloop.cta_tile_m
        kernel_extension = BlockScaledSwapAbFc12Extension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_pointer=fc1_done_counter.iterator,
            fc2_spin_threshold=fc2_spin_threshold,
            fc1_ready_counter_pointer=None,
        )
        optional_epilogue_args = GatedActEpilogueArgs(
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            topk_scores=topk_scores,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx == self.scheduler_warp_id:
            work_tile = scheduler.gen_next_work()
            while work_tile.is_valid_tile:
                scheduler.publish_work(kernel_extension.prepare_work_tile(work_tile))
                work_tile = scheduler.gen_next_work()
            scheduler.publish_work(work_tile)
            scheduler.produce_tail()

        if warp_idx == self.tma_a_warp_id:
            sched_consumer = scheduler.make_consumer()
            self._mainloop.run_tma_a(
                fc1_tma_a_tensor=fc1_tma_a_tensor,
                fc1_tma_a_atom=fc1_tma_a_atom,
                fc1_tma_sfa_tensor=fc1_tma_sfa_tensor,
                fc1_tma_sfa_atom=fc1_tma_sfa_atom,
                fc2_tma_a_tensor=fc2_tma_a_tensor,
                fc2_tma_a_atom=fc2_tma_a_atom,
                fc2_tma_sfa_tensor=fc2_tma_sfa_tensor,
                fc2_tma_sfa_atom=fc2_tma_sfa_atom,
                ab_pipeline=ab_pipeline,
                ab_pipeline_state=tma_a_pipeline_state,
                sched_consumer=sched_consumer,
                kernel_extension=kernel_extension,
            )

        if warp_idx == self.tma_b_warp_id:
            sched_consumer = scheduler.make_consumer()
            self._mainloop.run_tma_b(
                fc1_tma_b_tensor=fc1_tma_b_tensor,
                fc1_tma_b_atom_or_pair=fc1_tma_b_atom_or_pair,
                fc1_tma_sfb_tensor=fc1_tma_sfb_tensor,
                fc1_tma_sfb_atom_or_pair=fc1_tma_sfb_atom_or_pair,
                fc2_tma_b_tensor=fc2_tma_b_tensor,
                fc2_tma_b_atom_or_pair=fc2_tma_b_atom_or_pair,
                fc2_tma_sfb_tensor=fc2_tma_sfb_tensor,
                fc2_tma_sfb_atom_or_pair=fc2_tma_sfb_atom_or_pair,
                ab_pipeline=ab_pipeline,
                ab_pipeline_state=tma_b_pipeline_state,
                sched_consumer=sched_consumer,
                kernel_extension=kernel_extension,
                fc1_done_counter_pointer=fc1_done_counter.iterator,
                fc2_spin_threshold=fc2_spin_threshold,
            )

        if warp_idx == self.mma_warp_id:
            sched_consumer = scheduler.make_consumer()
            self._mainloop.run_mma(
                tmem_allocator=tmem_allocator,
                ab_pipeline=ab_pipeline,
                ab_pipeline_state=mma_pipeline_state,
                acc_pipeline=acc_pipeline,
                sched_consumer=sched_consumer,
            )

        if warp_idx < len(self.epilogue_warp_ids):
            sched_consumer = scheduler.make_consumer()
            tmem_allocator.allocate(self._mainloop.num_tmem_alloc_cols)
            tmem_allocator.wait_for_alloc()
            tmem_pointer = tmem_allocator.retrieve_ptr(self.acc_dtype)
            self.epilogue.run(
                self._smem_workspace,
                smem_base,
                tmem_pointer,
                acc_pipeline,
                sched_consumer,
                kernel_extension,
                fc1_output_tma_atom,
                fc1_output_tma_tensor,
                fc1_output_sf,
                fc2_output_tma_atom,
                fc2_output_tma_tensor,
                fc2_output,
                fc1_done_counter,
                thread_idx,
                None,
                None,
                None,
                None,
                optional_epilogue_args,
            )
            tmem_allocator.relinquish_alloc_permit()
            tmem_allocator.free(
                tmem_allocator.retrieve_ptr(self.acc_dtype), self._mainloop.num_tmem_alloc_cols
            )
