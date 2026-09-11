# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Local FC12 epilogue with metadata-based FC2 unrouting."""

import dataclasses
from typing import List, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64

from .....api import ImplDesc, OptionalRequirement, ProblemDesc
from .....helpers.cute_py_helpers import tcgen05_block_scaled_acc_dtype
from .....helpers.flag_batch import make_flag_batch_tracker
from .....helpers.smem_workspace import SmemRegion, SmemWorkspace
from .....quant_def import QuantKind
from ....function_mapping import FunctionMapping
from ....schedulers.base import SchedulerConsumer
from ....schedulers.fc12_mapping import BlockPhase, SwapAbFc12WorkTileInfo
from ..mega.block_scaled_swap_ab_fc12_epilogue import (
    GatedActEpilogueArgs,
    SwapABFc1Epilogue,
    make_fc2_redg_process_pipeline,
    make_fc2_stg_process_pipeline,
    make_fc2_ublk_process_pipeline,
)
from ..mega.block_scaled_swap_ab_fc12_epilogue import SwapABFc2Epilogue as _SharedSwapABFc2Epilogue
from ..mega.block_scaled_swap_ab_fc12_epilogue import (
    SwapABGatedActEpilogue as _SharedSwapABGatedActEpilogue,
)
from ..mega.block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension
from .local_routing import RoutingMetadata


class SwapABGatedActEpilogue(_SharedSwapABGatedActEpilogue):
    """BF16 Local MegaMoE epilogue.

    FC1 reuses the common register and staging implementation. FC2 always
    resolves a pool row through local routing metadata, either accumulating
    directly into the output or staging ``(token, topk_slot, hidden)`` for the
    standalone TopK reduction.
    """

    @classmethod
    def impl_desc_require(cls) -> dict[str, object]:
        return {
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "fc2_use_bulk": bool,
            "fc1_epi_flag_batch": int,
            "fc2_tma_stages": OptionalRequirement(int),
            "reduce_topk_in_kernel": OptionalRequirement(bool),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.acc_dtype = tcgen05_block_scaled_acc_dtype
        self.hidden_size = problem_desc["hidden_size"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.combine_format = problem_desc["combine_format"]
        self.gate_up_clamp = problem_desc["gate_up_clamp"]
        self.situ_beta, self.situ_linear_beta = self._resolve_situ_betas(problem_desc)
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.fc2_use_bulk = impl_desc["fc2_use_bulk"]
        self.fc1_epi_flag_batch = impl_desc["fc1_epi_flag_batch"]
        self.reduce_topk_in_kernel = impl_desc.get("reduce_topk_in_kernel", False)

        if self.combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError("Local MegaMoE epilogue requires a BF16 combine format.")

        self.fc1_output_dtype = self.quant_kind.activation_dtype
        self.fc1_output_sf_dtype = self.quant_kind.sf_dtype
        self.sf_vec_size = self.quant_kind.sf_vec_size
        self.needs_pair_amax_exchange = self.sf_vec_size > self._EpilogueFc1IntermediateDownPerWarp
        self.fc2_use_tma = False
        self.fc2_use_ublk = self.fc2_use_bulk
        self.reduce_topk_in_epilogue = self.reduce_topk_in_kernel
        if not 1 <= self.fc1_epi_flag_batch <= self._EpilogueWarpCnt:
            raise ValueError(
                f"Asynchronous FC1 flag batch size must be in [1, {self._EpilogueWarpCnt}]."
            )
        self.cluster_tile_intermediate_downproj = (
            self._EpilogueFc1IntermediateDownTileSize * self.cluster_shape_mn[0]
        )

        atom_thr_size = 2 if self.use_2cta_instrs else 1
        self.cta_tile_m = self._EpilogueFc2HiddenTileSize
        self.cta_tile_n = self.mma_tiler_mnk[1]
        self.cta_tile_k = self.mma_tiler_mnk[2]
        assert self.mma_tiler_mnk[0] // atom_thr_size == self.cta_tile_m
        assert self.cta_tile_n % self._EpilogueTokenTileSize == 0
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

        self.fc2_hidden_needs_predicate = not (
            isinstance(self.hidden_size, int)
            and self.hidden_size % (self.cta_tile_m * self.cluster_shape_mn[0]) == 0
        )
        self.intermediate_downproj = (
            self.intermediate_gateup_size // 2
            if isinstance(self.intermediate_gateup_size, int)
            else None
        )
        self.subtile_cnt = self.cta_tile_n // self._EpilogueTokenTileSize

        self.fc1_staging_stage_bytes = (
            self._EpilogueTokenTileSize
            * self._EpilogueFc1IntermediateDownTileSize
            * self.fc1_output_dtype.width
            // 8
        )
        self.fc1_staging_bytes = self.subtile_cnt * self.fc1_staging_stage_bytes
        self.fc1_amax_token_chunks = self._EpilogueTokenTileSize // 32
        self.fc1_amax_slot_count = (
            self.fc1_amax_token_chunks * self._EpilogueWarpCnt * 32
            if self.needs_pair_amax_exchange
            else 0
        )
        if self.fc1_amax_slot_count * 4 > self.fc1_staging_stage_bytes:
            raise ValueError("The paired amax exchange does not fit in one FC1 staging stage.")

        requested_fc2_tma_stages = impl_desc.get("fc2_tma_stages")
        if (
            requested_fc2_tma_stages is not None
            and not 1 <= requested_fc2_tma_stages <= self.subtile_cnt
        ):
            raise ValueError(
                f"fc2_tma_stages must be in [1, {self.subtile_cnt}], got {requested_fc2_tma_stages}."
            )
        if self.fc2_use_bulk:
            single_stage_region = self._make_fc2_single_stage_region()
            if single_stage_region.nbytes % 16 != 0:
                raise ValueError("Each FC2 UBLK staging stage must occupy a multiple of 16 bytes.")
            if requested_fc2_tma_stages is not None:
                self.fc2_tma_stages = requested_fc2_tma_stages
            else:
                bf16_baseline_stages = min(2, self.subtile_cnt)
                bf16_stage_bytes = (
                    self._EpilogueTokenTileSize
                    * self._EpilogueFc2HiddenTileSize
                    * cutlass.BFloat16.width
                    // 8
                )
                available_staging_bytes = max(
                    self.fc1_staging_bytes, bf16_baseline_stages * bf16_stage_bytes
                )
                self.fc2_tma_stages = min(
                    self.subtile_cnt, available_staging_bytes // single_stage_region.nbytes
                )
            self.fc2_staging_spec: Optional[SmemRegion] = self._make_fc2_staging_region(
                self.fc2_tma_stages
            )
        else:
            self.fc2_tma_stages = 0
            self.fc2_staging_spec = None

    def prepare_tma_store_params(
        self, fc1_output_template: cute.Tensor
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        fc1_operation = cpasync.CopyBulkTensorTileS2GOp()
        fc1_smem_layout = self.fc1_staged_smem_layout(1, without_stage_mode=True)
        fc1_tile = (self._EpilogueTokenTileSize, self._EpilogueFc1IntermediateDownTileSize)
        return cpasync.make_tiled_tma_atom(
            fc1_operation,
            fc1_output_template,
            fc1_smem_layout,
            fc1_tile,
        )

    @cute.jit
    def run(
        self,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        tmem_ptr: cute.Pointer,
        acc_pipeline,
        sched_consumer: SchedulerConsumer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        tma_atom_fc1_output: cute.CopyAtom,
        fc1_output: cute.Tensor,
        fc1_output_sf: cute.Tensor,
        fc2_output: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        tidx: cutlass.Int32,
        routing_metadata: cute.Tensor,
        optional_epi_args: Optional[GatedActEpilogueArgs] = None,
    ):
        if cutlass.const_expr(not smem_workspace.finalized):
            raise RuntimeError("SwapABGatedActEpilogue.run requires a finalized SmemWorkspace.")
        if cutlass.const_expr(optional_epi_args is None):
            optional_epi_args = GatedActEpilogueArgs(
                fc1_alpha=None, fc2_alpha=None, fc1_norm_const=None, topk_scores=None
            )
        if cutlass.const_expr(routing_metadata is None):
            raise ValueError("Local MegaMoE epilogue requires routing metadata.")

        fc1_staging_pointer = smem_workspace.ptr(self.fc1_staging_region, smem_base)
        if cutlass.const_expr(self.fc2_tma_stages > 0):
            fc2_smem_tensor = smem_workspace.tensor(self.fc2_staging_region, smem_base)
        else:
            fc2_smem_tensor = None
        tmem_acc = cute.make_tensor(
            cute.recast_ptr(tmem_ptr, dtype=cutlass.Float32),
            cute.make_layout(self.accumulator_shape, stride=self.accumulator_stride),
        )

        fc1_epi = SwapABFc1Epilogue(
            self,
            tidx,
            fc1_staging_pointer,
            kernel_extension,
            tma_atom_fc1_output,
            fc1_output,
            fc1_output_sf,
            fc1_done_counter,
            optional_epi_args,
        )
        fc2_epi = SwapABFc2Epilogue(
            self,
            tidx,
            fc2_smem_tensor,
            fc2_output,
            routing_metadata,
            optional_epi_args,
        )

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_accumulator_pipeline_stages
        )
        wait_only_named_barrier = self.epilogue_sync_barrier()
        work_tile_info = sched_consumer.consume_work()
        flag_tracker = make_flag_batch_tracker(
            True,
            flag_address=Int64(0),
            accumulated_flags=cutlass.Int32(0),
            phase=cutlass.Int32(work_tile_info.phase),
            thread_idx=tidx % (self._EpilogueWarpCnt * 32),
        )

        while work_tile_info.is_valid_tile:
            tmem_acc_current = tmem_acc[None, None, acc_consumer_state.index]
            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                fc1_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                )
            else:
                fc2_epi(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor=tmem_acc_current,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                )
            prev_work_tile_info = work_tile_info
            cur_was_linear1 = prev_work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
            acc_consumer_state.advance()
            work_tile_info = sched_consumer.consume_work()

            cute.arch.cp_async_bulk_wait_group(0)
            wait_only_named_barrier.arrive_and_wait()
            if cur_was_linear1:
                flag_tracker = fc1_epi.signal_fc1_done(
                    prev_work_tile_info, work_tile_info, flag_tracker
                )
            else:
                flag_tracker = fc2_epi.advance_fc1_flag_phase(work_tile_info, flag_tracker)
        flag_tracker.fire()


class SwapABFc2Epilogue(_SharedSwapABFc2Epilogue):
    """Device-side BF16 FC2 epilogue with local metadata addressing."""

    def __init__(
        self,
        base: SwapABGatedActEpilogue,
        tidx: cutlass.Int32,
        smem_tensor: Optional[cute.Tensor],
        fc2_output: cute.Tensor,
        routing_metadata: cute.Tensor,
        optional_epi_args: GatedActEpilogueArgs,
    ):
        self.base = base
        self.tidx = tidx % (base._EpilogueWarpCnt * 32)
        self.warp_idx = self.tidx // 32
        self.lane_idx = self.tidx % 32
        self.fc2_output = fc2_output
        self.routing_metadata = routing_metadata
        self.optional_epi_args = optional_epi_args
        if cutlass.const_expr(base.fc2_use_ublk):
            self.smem_tensor = smem_tensor
            self.process_pipeline = make_fc2_ublk_process_pipeline(
                combine_format=base.combine_format,
                cta_token_tile_size=base.cta_tile_n,
                cta_hidden_tile_size=base.cta_tile_m,
            )
        else:
            self.smem_tensor = None
            if cutlass.const_expr(base.reduce_topk_in_epilogue):
                self.process_pipeline = make_fc2_redg_process_pipeline(
                    combine_format=base.combine_format,
                    cta_token_tile_size=base.cta_tile_n,
                    cta_hidden_tile_size=base.cta_tile_m,
                )
            else:
                self.process_pipeline = make_fc2_stg_process_pipeline(
                    combine_format=base.combine_format,
                    cta_token_tile_size=base.cta_tile_n,
                    cta_hidden_tile_size=base.cta_tile_m,
                )
        self._freeze()

    def __extract_mlir_values__(self) -> List[ir.Value]:
        return []

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "SwapABFc2Epilogue":
        assert len(values) == 0
        return self

    @cute.jit
    def advance_fc1_flag_phase(self, next_work_tile_info, flag_tracker):
        return flag_tracker.accumulate(
            next_work_tile_info.phase,
            1,
            Int64(0),
            True,
        )

    @cute.jit
    def _make_output_router(self, work_tile_info: SwapAbFc12WorkTileInfo) -> "Fc2OutputRouter":
        task_tile_data_row_start = (
            work_tile_info.cumulative_data_physical_row
            + work_tile_info.tile_n_idx * cutlass.Int32(self.cta_tile_n)
        )
        hidden_base_this_cta_tile = work_tile_info.tile_m_idx * cutlass.Int32(self.cta_tile_m)
        valid_hidden_this_cta_tile = (
            cutlass.Int32(self.fc2_output.shape[2]) - hidden_base_this_cta_tile
        )
        if valid_hidden_this_cta_tile < 0:
            valid_hidden_this_cta_tile = 0
        if valid_hidden_this_cta_tile > self._EpilogueFc2HiddenTileSize:
            valid_hidden_this_cta_tile = self._EpilogueFc2HiddenTileSize

        return Fc2OutputRouter(
            routing_metadata=cute.domain_offset((task_tile_data_row_start,), self.routing_metadata),
            output=self.fc2_output,
            hidden_base_this_cta_tile=hidden_base_this_cta_tile,
            valid_tokens_this_cta_tile=work_tile_info.valid_tokens_in_cta_tile,
            valid_hidden_this_cta_tile=valid_hidden_this_cta_tile,
            reduce_topk_in_epilogue=self.reduce_topk_in_epilogue,
            data_mapping=self.process_pipeline.store_out_mapping,
            epi_tid=self.tidx,
        ).prefetch()


@dataclasses.dataclass(frozen=True)
class Fc2OutputRouter:
    """Resolve FC2 pool rows to local token/top-k destinations."""

    routing_metadata: cute.Tensor
    output: cute.Tensor
    hidden_base_this_cta_tile: Union[cutlass.Int32, int]
    valid_tokens_this_cta_tile: cutlass.Int32
    valid_hidden_this_cta_tile: Union[cutlass.Int32, int]
    reduce_topk_in_epilogue: bool
    data_mapping: FunctionMapping
    epi_tid: cutlass.Int32
    dst_ptrs: Optional[cute.Tensor] = None
    valid: Optional[cute.Tensor] = None

    @property
    def data_output(self) -> cute.Tensor:
        return self.output

    @cute.jit
    def prefetch(self) -> "Fc2OutputRouter":
        copy_iters: cutlass.Constexpr[int] = self.data_mapping.domain.axis_size("iter_idx")
        valid = cute.make_rmem_tensor((copy_iters,), cutlass.Int32)
        dst_ptrs = cute.make_rmem_tensor((copy_iters,), cutlass.Int64)

        for iter_idx in cutlass.range_constexpr(copy_iters):
            coord = self.data_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=iter_idx)
            token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
            hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])
            valid[iter_idx] = cutlass.Int32(0)
            dst_ptrs[iter_idx] = cutlass.Int64(0)

            token_valid = token_in_tile < self.valid_tokens_this_cta_tile
            hidden_valid = hidden_in_tile < cutlass.Int32(self.valid_hidden_this_cta_tile)
            if token_valid and hidden_valid:
                valid[iter_idx] = cutlass.Int32(1)
                metadata = RoutingMetadata.load(
                    self.routing_metadata.iterator.toint()
                    + Int64(token_in_tile) * Int64(RoutingMetadata.nbytes)
                )
                dst_topk = (
                    cutlass.Int32(0)
                    if cutlass.const_expr(self.reduce_topk_in_epilogue)
                    else metadata.topk_slot
                )
                dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                dst_ptrs[iter_idx] = cute.domain_offset(
                    (Int64(metadata.token), dst_topk, dst_hidden), self.output
                ).iterator.toint()

        return dataclasses.replace(self, dst_ptrs=dst_ptrs, valid=valid)

    @cute.jit
    def get_data_dst(
        self, iter_idx: Union[int, cutlass.Int32]
    ) -> Tuple[cute.Pointer, cutlass.Int32]:
        ptr = cute.make_ptr(
            self.output.element_type,
            self.dst_ptrs[iter_idx],
            AddressSpace.gmem,
            assumed_align=32,
        )
        return ptr, self.valid[iter_idx]


__all__ = [
    "Fc2OutputRouter",
    "GatedActEpilogueArgs",
    "SwapABGatedActEpilogue",
]
