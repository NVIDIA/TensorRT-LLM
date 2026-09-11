# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata-push routing and fused non-deterministic token communication."""

import dataclasses
import os
from typing import Callable, ClassVar, Literal, Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

from ...api import ImplDesc, KernelComponent, OptionalRequirement, ProblemDesc
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.dsl_helpers import mark_alignment, smem_exclusive_prefix
from ...helpers.flag_batch import make_flag_batch_tracker
from ...helpers.iket_compat import iket
from ...helpers.ptx_helpers import (
    cp_async_bulk_s2g,
    cp_reduce_async_bulk_add_bf16_s2g,
    cp_reduce_async_bulk_add_u32_s2g,
    nanosleep,
    read_clock64,
    red_add_relaxed_sys_s32,
    stg_b64,
    stg_f32,
    tma_load_1d,
)
from ...helpers.smem_workspace import SmemWorkspace
from ...helpers.software_sync import NvlinkBarrier
from ...helpers.utils import ceil_div, round_up
from ...quant_def import CombineFormat, QuantKind
from ..token_protocol import TokenSrcMetadata
from .symmetric_buffer import SymmetricBufferDevice

TokenBackMode = Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
TokenBackScheduleMode = Literal["static", "atomic_counter"]


@dataclasses.dataclass(frozen=True)
class TokenCommArgs:
    """Device views materialized inside the fused kernel region.

    ``pre_reduced_activation`` is the per-topk combine staging plane: TokenComm's own symmetric region, except
    under in-kernel top-k reduction where it degenerates to a view of the caller's 2D output. That REDG
    accumulates, so in that mode the incoming content is the caller's accumulation base -- zero, or a
    shared-expert result.
    """

    activation: cute.Tensor
    activation_sf: cute.Tensor
    pre_reduced_activation: cute.Tensor
    pre_reduced_activation_sf: Optional[cute.Tensor]
    peer_rank_ptr_mapper: SymmetricBufferDevice

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "TokenCommArgs":
        if values:
            raise ValueError(f"TokenCommArgs expected no MLIR values, got {len(values)}.")
        return self


@dataclasses.dataclass(frozen=True)
class _SortedElement:
    flat_topk_index: Int32
    topk_score: Optional[cutlass.Float32]

    def pack(self) -> Union[Int64, Int32]:
        if cutlass.const_expr(self.topk_score is None):
            return self.flat_topk_index
        scratch = cute.make_rmem_tensor((2,), cutlass.Int32)
        scratch[0] = self.flat_topk_index
        cute.recast_tensor(scratch, cutlass.Float32)[1] = self.topk_score
        return cute.recast_tensor(scratch, cutlass.Int64)[0]

    @classmethod
    def from_packed(cls, packed: Union[Int64, Int32]) -> "_SortedElement":
        if cutlass.const_expr(type(packed).width == 32):
            return cls(flat_topk_index=Int32(packed), topk_score=None)
        scratch = cute.make_rmem_tensor((2,), cutlass.Int32)
        cute.recast_tensor(scratch, cutlass.Int64)[0] = packed
        return cls(
            flat_topk_index=scratch[0], topk_score=cute.recast_tensor(scratch, cutlass.Float32)[1]
        )


@cute.jit
def _copy_atom(dtype, num_bits_per_copy: int):
    return cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=num_bits_per_copy
    )


class _MetadataPushRouter(KernelComponent):
    """Sort and push routing metadata into each destination rank's final pool."""

    router_smem_limit_bytes: ClassVar[int] = 227 * 1024
    router_warps_per_cta: ClassVar[int] = 16

    sizes_by_rank_region = "nvlink.token_comm.sizes_by_rank"
    sizes_region = "nvlink.token_comm.sizes"
    sizes_ready_region = "nvlink.token_comm.sizes_ready"
    metadata_ready_region = "nvlink.token_comm.metadata_ready"
    sorted_metadata_region = "nvlink.token_comm.sorted_metadata"
    sorted_scores_region = "nvlink.token_comm.sorted_scores"
    pool_expert_base_region = "nvlink.token_comm.pool_expert_base"
    token_src_metadata_region = "nvlink.token_comm.token_src_metadata"
    fc1_topk_scores_region = "nvlink.token_comm.fc1_topk_scores"
    source_expert_base_region = "nvlink.token_comm.source_expert_base"
    push_destination_base_region = "nvlink.token_comm.push_destination_base"
    sorted_metadata_ready_region = "nvlink.token_comm.sorted_metadata_ready"
    push_table_ready_region = "nvlink.token_comm.push_table_ready"
    router_size_counter_region = "nvlink.token_comm.router_size_counter"
    router_histogram_done_region = "nvlink.token_comm.router_histogram_done"
    source_base_ready_region = "nvlink.token_comm.source_base_ready"

    router_data_histogram_region = "nvlink.token_comm.router_smem.data_histogram"
    router_data_prefix_region = "nvlink.token_comm.router_smem.data_prefix"
    router_data_warp_totals_region = "nvlink.token_comm.router_smem.data_warp_totals"
    router_data_sorted_region = "nvlink.token_comm.router_smem.data_sorted"
    router_data_base_region = "nvlink.token_comm.router_smem.data_base"
    router_helper_size_matrix_region = "nvlink.token_comm.router_smem.helper_size_matrix"
    router_helper_totals_region = "nvlink.token_comm.router_smem.helper_totals"
    router_helper_prefix_region = "nvlink.token_comm.router_smem.helper_prefix"
    router_helper_warp_totals_region = "nvlink.token_comm.router_smem.helper_warp_totals"
    router_helper_load_mbarrier_region = "nvlink.token_comm.router_smem.helper_load_mbarrier"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "world_size": int,
            "expert_count": int,
            "topk": int,
            "max_tokens_per_rank": int,
            "apply_topk_at_fc1": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "token_padding_block": int,
            "promised_launchable_sm_count": int,
            "router_smem_limit_bytes": OptionalRequirement(int),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.world_size = problem_desc["world_size"]
        self.expert_count = problem_desc["expert_count"]
        self.topk = problem_desc["topk"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.token_padding_block = impl_desc["token_padding_block"]
        self.promised_launchable_sm_count = impl_desc["promised_launchable_sm_count"]
        self.router_smem_limit_bytes = impl_desc.get("router_smem_limit_bytes", 227 * 1024)

        self._validate_router_configuration()
        self.expert_count_padded = round_up(self.expert_count, 4)
        self.expert_count_with_trash = self.expert_count_padded + 1
        self.router_elements_per_lane, self.router_data_cta_count = (
            self._router_launch_configuration()
        )
        self.router_tokens_per_cta = self.router_elements_per_lane * self.router_warps_per_cta * 32
        self.router_push_cta_count = ceil_div(self.expert_count, self.router_warps_per_cta)
        self.router_grid_cta_count = max(self.router_data_cta_count + 1, self.router_push_cta_count)
        if self.router_grid_cta_count > self.promised_launchable_sm_count:
            raise ValueError(
                "Router grid exceeds promised_launchable_sm_count; "
                "all metadata-push CTAs must be concurrently resident."
            )
        self.worst_case_token_count = self.worst_case_padded_tokens(self.token_padding_block)
        self._router_smem_workspace = self._build_router_smem_workspace()

        self._device_workspace = None
        self._peer_rank_ptr_mapper = None
        self._router_local_rank = None
        self._router_thread_idx = None
        self._router_linear_cta_idx = None
        self._router_grid_thread_idx = None
        self._router_warp_idx = None
        self._router_lane_idx = None

    def _validate_router_configuration(self) -> None:
        positive_fields = (
            "world_size",
            "expert_count",
            "topk",
            "max_tokens_per_rank",
            "token_padding_block",
            "promised_launchable_sm_count",
            "router_smem_limit_bytes",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")
        if self.expert_count % self.world_size != 0:
            raise ValueError(
                f"expert_count must be divisible by world_size, got {self.expert_count} and {self.world_size}."
            )
        if self.expert_count > 16384:
            raise NotImplementedError("TokenComm supports at most 16384 global experts.")
        if self.topk > self.expert_count:
            raise ValueError(
                f"topk must not exceed expert_count, got {self.topk} and {self.expert_count}."
            )

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.world_size

    def worst_case_padded_tokens(self, block: int) -> int:
        source_token_capacity = self.world_size * self.max_tokens_per_rank
        routes_per_source_token = min(self.topk, self.experts_per_rank)
        route_capacity = source_token_capacity * routes_per_source_token
        active_expert_capacity = min(self.experts_per_rank, route_capacity)
        route_budget_blocks = (
            active_expert_capacity + (route_capacity - active_expert_capacity) // block
        )
        expert_bound_blocks = active_expert_capacity * int(ceil_div(source_token_capacity, block))
        return min(route_budget_blocks, expert_bound_blocks) * block

    def _router_launch_configuration(self) -> Tuple[int, int]:
        def next_power_of_two(value: int) -> int:
            return 1 << (max(value, 1) - 1).bit_length()

        routed_token_capacity = self.max_tokens_per_rank * self.topk
        minimum_cta_capacity = 2048
        maximum_cta_capacity = 16384
        maximum_data_cta_count = 128
        maximum_supported_tokens = maximum_cta_capacity * maximum_data_cta_count
        if routed_token_capacity > maximum_supported_tokens:
            raise NotImplementedError(
                f"The router supports at most {maximum_supported_tokens} routed tokens per rank."
            )
        cta_capacity = min(
            maximum_cta_capacity,
            next_power_of_two(max(routed_token_capacity, minimum_cta_capacity)),
        )
        elements_per_lane = cta_capacity // (self.router_warps_per_cta * 32)
        data_cta_count = ceil_div(routed_token_capacity, cta_capacity)
        return elements_per_lane, data_cta_count

    def _build_router_smem_workspace(self) -> SmemWorkspace:
        workspace = SmemWorkspace()
        workspace.register_mbarrier(self.router_helper_load_mbarrier_region, 1)
        overlay = workspace.create_overlay("nvlink.token_comm.router_smem.role")
        data_lifetime = overlay.add_lifetime("data_cta")
        data_lifetime.register_tensor(
            self.router_data_histogram_region, cutlass.Int32, (self.expert_count_with_trash,)
        )
        data_lifetime.register_tensor(
            self.router_data_prefix_region,
            cutlass.Int32,
            (self.expert_count_with_trash,),
            byte_alignment=16,
        )
        data_lifetime.register_tensor(
            self.router_data_warp_totals_region, cutlass.Int32, (self.router_warps_per_cta,)
        )
        data_lifetime.register_tensor(
            self.router_data_sorted_region,
            (cutlass.Int64 if self.apply_topk_at_fc1 else cutlass.Int32),
            (self.router_tokens_per_cta,),
            byte_alignment=16,
        )
        if self.router_data_cta_count > 1:
            data_lifetime.register_tensor(
                self.router_data_base_region,
                cutlass.Int32,
                (self.expert_count_padded,),
                byte_alignment=16,
            )

        helper_lifetime = overlay.add_lifetime("helper_cta")
        helper_lifetime.register_tensor(
            self.router_helper_size_matrix_region,
            cutlass.Int32,
            (self.world_size, self.expert_count_padded),
            stride=(self.expert_count_padded, 1),
            byte_alignment=16,
        )
        helper_lifetime.register_tensor(
            self.router_helper_totals_region,
            cutlass.Int32,
            (self.expert_count_padded,),
            byte_alignment=16,
        )
        helper_lifetime.register_tensor(
            self.router_helper_prefix_region,
            cutlass.Int32,
            (self.expert_count_padded,),
            byte_alignment=16,
        )
        helper_lifetime.register_tensor(
            self.router_helper_warp_totals_region,
            cutlass.Int32,
            (self.router_warps_per_cta,),
            byte_alignment=16,
        )
        workspace.finalize(max_bytes=self.router_smem_limit_bytes)
        return workspace

    @property
    def router_smem_workspace(self) -> SmemWorkspace:
        return self._router_smem_workspace

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        """Register router-private state and Router-to-Main outputs."""
        self._register_router_workspace(workspace)

    def _register_router_workspace(self, workspace: DeviceWorkspace) -> None:
        maximum_routed_tokens = self.max_tokens_per_rank * self.topk
        workspace.register(
            self.sizes_by_rank_region,
            cutlass.Int32,
            (self.world_size, self.expert_count_padded),
            buffer_space="shared",
            stride=(self.expert_count_padded, 1),
        )
        workspace.register(
            self.sizes_region,
            cutlass.Int32,
            (self.expert_count_padded,),
            buffer_space="shared",
            reset="tail_reset",
        )
        workspace.register(
            self.sizes_ready_region, cutlass.Int32, (1,), buffer_space="shared", reset="tail_reset"
        )
        workspace.register(
            self.metadata_ready_region,
            cutlass.Int32,
            (1,),
            buffer_space="shared",
            reset="tail_reset",
        )
        workspace.register(
            self.sorted_metadata_region,
            cutlass.Int64,
            (maximum_routed_tokens,),
            buffer_space="local",
        )
        if self.apply_topk_at_fc1:
            workspace.register(
                self.sorted_scores_region,
                cutlass.Float32,
                (maximum_routed_tokens,),
                buffer_space="local",
            )
        workspace.register(
            self.token_src_metadata_region,
            cutlass.Int64,
            (self.worst_case_token_count,),
            buffer_space="shared",
            byte_alignment=16,
        )
        if self.apply_topk_at_fc1:
            workspace.register(
                self.fc1_topk_scores_region,
                cutlass.Float32,
                (self.worst_case_token_count,),
                buffer_space="shared",
            )
        workspace.register(
            self.pool_expert_base_region,
            cutlass.Int32,
            (self.experts_per_rank,),
            buffer_space="local",
        )
        workspace.register(
            self.source_expert_base_region,
            cutlass.Int32,
            (self.expert_count_padded,),
            buffer_space="local",
        )
        workspace.register(
            self.push_destination_base_region,
            cutlass.Int32,
            (self.expert_count_padded,),
            buffer_space="local",
        )
        workspace.register(
            self.sorted_metadata_ready_region,
            cutlass.Int32,
            (1,),
            buffer_space="local",
            reset="tail_reset",
        )
        workspace.register(
            self.push_table_ready_region,
            cutlass.Int32,
            (1,),
            buffer_space="local",
            reset="tail_reset",
        )
        if self.router_data_cta_count > 1:
            workspace.register(
                self.router_size_counter_region,
                cutlass.Int32,
                (self.expert_count_with_trash,),
                buffer_space="local",
                reset="tail_reset",
            )
            workspace.register(
                self.router_histogram_done_region,
                cutlass.Int32,
                (1,),
                buffer_space="local",
                reset="tail_reset",
            )
            workspace.register(
                self.source_base_ready_region,
                cutlass.Int32,
                (1,),
                buffer_space="local",
                reset="tail_reset",
            )

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "_MetadataPushRouter":
        if values:
            raise ValueError("_MetadataPushRouter carries no MLIR values.")
        return self

    @cute.jit
    def launch_router(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper_host,
        device_workspace: DeviceWorkspace,
        stream: cuda.CUstream,
    ) -> None:
        """Launch counting-sort DATA, size-exchange HELPER, and metadata PUSH roles."""
        if cutlass.const_expr(self.apply_topk_at_fc1 and topk_scores is None):
            raise ValueError("apply_topk_at_fc1 requires router topk_scores.")
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_object()
        self._router_kernel(
            topk_indices,
            topk_scores,
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper,
            device_workspace,
        ).launch(
            grid=[self.router_grid_cta_count, 1, 1],
            block=[self.router_warps_per_cta * 32, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def _router_kernel(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        device_workspace: DeviceWorkspace,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        linear_cta_idx, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_launch_dependents()
        block_thread_count = self.router_warps_per_cta * 32
        grid_thread_idx = thread_idx + linear_cta_idx * block_thread_count
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        lane_idx = thread_idx % Int32(32)

        storage_type = self._router_smem_workspace.storage_class()
        smem_allocator = cutlass.utils.SmemAllocator()
        storage = smem_allocator.allocate(storage_type)
        smem_base = storage.buffer.data_ptr()

        device_workspace.assign_device_members(local_workspace, shared_workspace)
        self._device_workspace = device_workspace
        self._peer_rank_ptr_mapper = peer_rank_ptr_mapper
        self._router_local_rank = local_rank
        self._router_thread_idx = thread_idx
        self._router_linear_cta_idx = linear_cta_idx
        self._router_grid_thread_idx = grid_thread_idx
        self._router_warp_idx = warp_idx
        self._router_lane_idx = lane_idx

        if cutlass.const_expr(self.router_data_cta_count == 1):
            self._router_single_cta(topk_indices, topk_scores, smem_base)
        else:
            self._router_multiple_ctas(topk_indices, topk_scores, smem_base)
        if linear_cta_idx < Int32(self.router_push_cta_count):
            self._router_push_metadata()

        device_workspace.remove_device_members()
        self._device_workspace = None
        self._peer_rank_ptr_mapper = None
        self._router_local_rank = None
        self._router_thread_idx = None
        self._router_linear_cta_idx = None
        self._router_grid_thread_idx = None
        self._router_warp_idx = None
        self._router_lane_idx = None

    @cute.jit
    def _router_single_cta(
        self, topk_indices: cute.Tensor, topk_scores: Optional[cute.Tensor], smem_base: cute.Pointer
    ) -> None:
        if self._router_linear_cta_idx < Int32(self.router_data_cta_count):
            block_thread_count = self.router_warps_per_cta * 32
            trash_bucket = self.expert_count_padded
            histogram = self._router_smem_workspace.tensor(
                self.router_data_histogram_region, smem_base
            )
            prefix = self._router_smem_workspace.tensor(self.router_data_prefix_region, smem_base)
            warp_totals = self._router_smem_workspace.tensor(
                self.router_data_warp_totals_region, smem_base
            )
            sorted_elements = self._router_smem_workspace.tensor(
                self.router_data_sorted_region, smem_base
            )

            zero_round_count = ceil_div(self.expert_count_with_trash, block_thread_count)
            for zero_round in cutlass.range_constexpr(zero_round_count):
                expert = Int32(zero_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_with_trash):
                    histogram[expert] = Int32(0)

            iket.range_push("router.histogram")
            expert_registers, score_registers = self._load_router_inputs(topk_indices, topk_scores)
            cute.arch.sync_threads()
            within_expert_indices = self._build_histogram(expert_registers, histogram)
            iket.range_pop()

            iket.range_push("router.prefix_and_publish")
            publish_sizes = self._broadcast_sizes_to_peers(
                cute.make_tensor(histogram.iterator, cute.make_layout((self.expert_count_padded,)))
            )
            total_valid_routes = smem_exclusive_prefix(
                cute.make_tensor(histogram.iterator, cute.make_layout((self.expert_count_padded,))),
                cute.make_tensor(prefix.iterator, cute.make_layout((self.expert_count_padded,))),
                warp_totals,
                block_thread_count,
                self._router_thread_idx,
                self._router_lane_idx,
                self._router_warp_idx,
            )
            if self._router_thread_idx == Int32(0):
                prefix[trash_bucket] = total_valid_routes
            source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
            expert_round_count = ceil_div(self.expert_count_padded, block_thread_count)
            for expert_round in cutlass.range_constexpr(expert_round_count):
                expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_padded):
                    source_expert_base[expert] = prefix[expert]
            cute.arch.sync_threads()
            publish_sizes()
            iket.range_pop()

            iket.range_push("router.sort")
            self._sort_router_elements(
                expert_registers,
                within_expert_indices,
                score_registers,
                sorted_elements,
                prefix,
                topk_indices.dtype,
            )
            cute.arch.sync_threads()
            iket.range_pop()
            iket.range_push("router.write_out")
            self._dump_contiguous_router_output(sorted_elements, total_valid_routes)
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.sorted_metadata_ready_region),
                    Int32(1),
                    sem="release",
                    scope="gpu",
                )
            iket.range_pop()
        elif self._router_linear_cta_idx == Int32(self.router_data_cta_count):
            self._router_helper_single_cta(smem_base)

    @cute.jit
    def _router_multiple_ctas(
        self, topk_indices: cute.Tensor, topk_scores: Optional[cute.Tensor], smem_base: cute.Pointer
    ) -> None:
        if self._router_linear_cta_idx < Int32(self.router_data_cta_count):
            block_thread_count = self.router_warps_per_cta * 32
            trash_bucket = self.expert_count_padded
            expert_round_count = ceil_div(self.expert_count_padded, block_thread_count)
            histogram = self._router_smem_workspace.tensor(
                self.router_data_histogram_region, smem_base
            )
            prefix = self._router_smem_workspace.tensor(self.router_data_prefix_region, smem_base)
            warp_totals = self._router_smem_workspace.tensor(
                self.router_data_warp_totals_region, smem_base
            )
            sorted_elements = self._router_smem_workspace.tensor(
                self.router_data_sorted_region, smem_base
            )
            dump_base = self._router_smem_workspace.tensor(self.router_data_base_region, smem_base)

            zero_round_count = ceil_div(self.expert_count_with_trash, block_thread_count)
            for zero_round in cutlass.range_constexpr(zero_round_count):
                expert = Int32(zero_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_with_trash):
                    histogram[expert] = Int32(0)

            iket.range_push("router.histogram")
            expert_registers, score_registers = self._load_router_inputs(topk_indices, topk_scores)
            cute.arch.sync_threads()
            within_expert_indices = self._build_histogram(expert_registers, histogram)
            iket.range_pop()

            iket.range_push("router.reserve_and_prefix")
            size_counter = self._device_workspace.ptr(self.router_size_counter_region)
            for expert_round in cutlass.range_constexpr(expert_round_count):
                expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_padded):
                    dump_base[expert] = Int32(
                        cute.arch.atomic_add(
                            size_counter + expert, histogram[expert], sem="relaxed", scope="gpu"
                        )
                    )
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.router_histogram_done_region),
                    Int32(1),
                    sem="release",
                    scope="gpu",
                )

            total_valid_routes = smem_exclusive_prefix(
                cute.make_tensor(histogram.iterator, cute.make_layout((self.expert_count_padded,))),
                cute.make_tensor(prefix.iterator, cute.make_layout((self.expert_count_padded,))),
                warp_totals,
                block_thread_count,
                self._router_thread_idx,
                self._router_lane_idx,
                self._router_warp_idx,
            )
            if self._router_thread_idx == Int32(0):
                prefix[trash_bucket] = total_valid_routes
            cute.arch.sync_threads()
            iket.range_pop()
            iket.range_push("router.sort")
            self._sort_router_elements(
                expert_registers,
                within_expert_indices,
                score_registers,
                sorted_elements,
                prefix,
                topk_indices.dtype,
            )
            cute.arch.sync_threads()
            iket.range_pop()

            iket.range_push("router.wait_source_base")
            source_base_ready = self._device_workspace.ptr(self.source_base_ready_region)
            if self._router_thread_idx == Int32(0):
                while cute.arch.load(source_base_ready, Int32, sem="acquire", scope="gpu") != Int32(
                    1
                ):
                    nanosleep(150)
            cute.arch.sync_threads()
            iket.range_pop()

            iket.range_push("router.write_out")
            source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
            for expert_round in cutlass.range_constexpr(expert_round_count):
                expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_padded):
                    dump_base[expert] = dump_base[expert] + source_expert_base[expert]
            cute.arch.sync_threads()
            self._dump_router_output_by_expert(histogram, prefix, dump_base, sorted_elements)
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.sorted_metadata_ready_region),
                    Int32(1),
                    sem="release",
                    scope="gpu",
                )
            iket.range_pop()
        elif self._router_linear_cta_idx == Int32(self.router_data_cta_count):
            self._router_helper_multiple_ctas(smem_base)

    @cute.jit
    def _router_helper_single_cta(self, smem_base: cute.Pointer) -> None:
        iket.range_push("router.compute_push_tables")
        self._compute_push_tables(smem_base)
        iket.range_pop()

    @cute.jit
    def _router_helper_multiple_ctas(self, smem_base: cute.Pointer) -> None:
        block_thread_count = self.router_warps_per_cta * 32
        totals = self._router_smem_workspace.tensor(self.router_helper_totals_region, smem_base)
        prefix = self._router_smem_workspace.tensor(self.router_helper_prefix_region, smem_base)
        warp_totals = self._router_smem_workspace.tensor(
            self.router_helper_warp_totals_region, smem_base
        )
        size_counter = self._device_workspace.tensor(self.router_size_counter_region)
        source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
        expert_round_count = ceil_div(self.expert_count_padded, block_thread_count)

        histogram_done = self._device_workspace.ptr(self.router_histogram_done_region)
        iket.range_push("router.wait_histogram")
        if self._router_thread_idx == Int32(0):
            while cute.arch.load(histogram_done, Int32, sem="acquire", scope="gpu") != Int32(
                self.router_data_cta_count
            ):
                nanosleep(150)
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("router.broadcast_sizes")
        for expert_round in cutlass.range_constexpr(expert_round_count):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if expert < Int32(self.expert_count_padded):
                totals[expert] = size_counter[expert]
        cute.arch.sync_threads()

        publish_sizes = self._broadcast_sizes_to_peers(totals)
        smem_exclusive_prefix(
            totals,
            prefix,
            warp_totals,
            block_thread_count,
            self._router_thread_idx,
            self._router_lane_idx,
            self._router_warp_idx,
        )
        for expert_round in cutlass.range_constexpr(expert_round_count):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if expert < Int32(self.expert_count_padded):
                source_expert_base[expert] = prefix[expert]
        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            cute.arch.atomic_add(
                self._device_workspace.ptr(self.source_base_ready_region),
                Int32(1),
                sem="release",
                scope="gpu",
            )
        publish_sizes()
        iket.range_pop()

        iket.range_push("router.compute_push_tables")
        self._compute_push_tables(smem_base)
        iket.range_pop()

    @cute.jit
    def _router_push_metadata(self) -> None:
        block_thread_count = self.router_warps_per_cta * 32
        sorted_metadata_ready = self._device_workspace.ptr(self.sorted_metadata_ready_region)
        push_table_ready = self._device_workspace.ptr(self.push_table_ready_region)
        if self._router_thread_idx == Int32(0):
            while cute.arch.load(sorted_metadata_ready, Int32, sem="acquire", scope="gpu") != Int32(
                self.router_data_cta_count
            ):
                nanosleep(150)
            while cute.arch.load(push_table_ready, Int32, sem="acquire", scope="gpu") != Int32(1):
                nanosleep(150)
        cute.arch.sync_threads()

        global_expert = (
            self._router_linear_cta_idx * Int32(self.router_warps_per_cta) + self._router_warp_idx
        )
        if global_expert < Int32(self.expert_count):
            sizes_by_rank = self._device_workspace.tensor(self.sizes_by_rank_region)
            source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
            push_destination_base = self._device_workspace.tensor(self.push_destination_base_region)
            route_count = sizes_by_rank[self._router_local_rank, global_expert]
            source_begin = source_expert_base[global_expert]
            destination_begin = push_destination_base[global_expert]
            destination_rank = global_expert // Int32(self.experts_per_rank)
            peer_offset = self._peer_rank_ptr_mapper.map(Int64(0), destination_rank, Int64(0))
            source_metadata = self._device_workspace.ptr(self.sorted_metadata_region)
            destination_metadata_address = (
                self._device_workspace.ptr(self.token_src_metadata_region).toint() + peer_offset
            )
            if cutlass.const_expr(self.apply_topk_at_fc1):
                source_scores = self._device_workspace.ptr(self.sorted_scores_region)
                destination_scores_address = (
                    self._device_workspace.ptr(self.fc1_topk_scores_region).toint() + peer_offset
                )
            route_round_count = (route_count + Int32(31)) // Int32(32)
            for route_round in cutlass.range(route_round_count, unroll=1):
                route = Int32(route_round) * Int32(32) + self._router_lane_idx
                if route < route_count:
                    source_position = source_begin + route
                    destination_position = destination_begin + route
                    metadata = cute.arch.load(source_metadata + source_position, cutlass.Int64)
                    stg_b64(
                        destination_metadata_address
                        + Int64(destination_position) * Int64(TokenSrcMetadata.nbytes),
                        metadata,
                    )
                    if cutlass.const_expr(self.apply_topk_at_fc1):
                        score = cute.arch.load(source_scores + source_position, cutlass.Float32)
                        stg_f32(
                            destination_scores_address + Int64(destination_position) * Int64(4),
                            score,
                        )

        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            cute.arch.fence_acq_rel_sys()
        # Keep notifier threads behind the leader's system fence.
        cute.arch.sync_threads()
        metadata_ready_address = self._device_workspace.ptr(self.metadata_ready_region).toint()
        rank_round_count = ceil_div(self.world_size, block_thread_count)
        for rank_round in cutlass.range_constexpr(rank_round_count):
            destination_rank = Int32(rank_round * block_thread_count) + self._router_thread_idx
            if destination_rank < Int32(self.world_size):
                red_add_relaxed_sys_s32(
                    self._peer_rank_ptr_mapper.map(
                        metadata_ready_address, destination_rank, Int64(0)
                    ),
                    Int32(1),
                )

    @cute.jit
    def _broadcast_sizes_to_peers(self, smem_expert_counts: cute.Tensor) -> Callable[[], None]:
        block_thread_count = self.router_warps_per_cta * 32
        row_bytes = Int32(self.expert_count_padded * 4)
        matrix_address = self._device_workspace.ptr(self.sizes_by_rank_region).toint()
        total_address = self._device_workspace.ptr(self.sizes_region).toint()
        rank_round_count = ceil_div(self.world_size, self.router_warps_per_cta)
        for rank_round in cutlass.range_constexpr(rank_round_count):
            destination_rank = self._router_warp_idx + Int32(rank_round * self.router_warps_per_cta)
            if destination_rank < Int32(self.world_size):
                peer_offset = self._peer_rank_ptr_mapper.map(Int64(0), destination_rank, Int64(0))
                destination_row_address = (
                    matrix_address
                    + peer_offset
                    + Int64(Int32(self._router_local_rank) * Int32(self.expert_count_padded))
                    * Int64(4)
                )
                destination_row = cute.make_ptr(
                    cutlass.Int32, destination_row_address, AddressSpace.gmem, assumed_align=16
                )
                destination_total = cute.make_ptr(
                    cutlass.Int32, total_address + peer_offset, AddressSpace.gmem, assumed_align=16
                )
                with cute.arch.elect_one():
                    cp_async_bulk_s2g(destination_row, smem_expert_counts.iterator, row_bytes)
                    cp_reduce_async_bulk_add_u32_s2g(
                        destination_total, smem_expert_counts.iterator, row_bytes
                    )
                    cute.arch.cp_async_bulk_commit_group()

        def finalize() -> None:
            cute.arch.cp_async_bulk_wait_group(0)
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.fence_acq_rel_sys()
            cute.arch.sync_threads()
            ready_address = self._device_workspace.ptr(self.sizes_ready_region).toint()
            ready_round_count = ceil_div(self.world_size, block_thread_count)
            for ready_round in cutlass.range_constexpr(ready_round_count):
                destination_rank = Int32(ready_round * block_thread_count) + self._router_thread_idx
                if destination_rank < Int32(self.world_size):
                    red_add_relaxed_sys_s32(
                        self._peer_rank_ptr_mapper.map(ready_address, destination_rank, Int64(0)),
                        Int32(1),
                    )

        return finalize

    @cute.jit
    def _compute_push_tables(self, smem_base: cute.Pointer) -> None:
        block_thread_count = self.router_warps_per_cta * 32
        owner_expert_begin = Int32(self._router_local_rank) * Int32(self.experts_per_rank)
        matrix_bytes = self.world_size * self.expert_count_padded * 4

        size_matrix = self._router_smem_workspace.tensor(
            self.router_helper_size_matrix_region, smem_base
        )
        padded_totals = self._router_smem_workspace.tensor(
            self.router_helper_totals_region, smem_base
        )
        prefix = self._router_smem_workspace.tensor(self.router_helper_prefix_region, smem_base)
        warp_totals = self._router_smem_workspace.tensor(
            self.router_helper_warp_totals_region, smem_base
        )
        load_mbarrier = self._router_smem_workspace.ptr(
            self.router_helper_load_mbarrier_region, smem_base
        )
        sizes = self._device_workspace.tensor(self.sizes_region)

        if self._router_thread_idx == Int32(0):
            cute.arch.mbarrier_init(load_mbarrier, 1)

        sizes_ready = self._device_workspace.ptr(self.sizes_ready_region)
        iket.range_push("router.wait_sizes_ready")
        if self._router_thread_idx == Int32(0):
            while cute.arch.load(sizes_ready, Int32, sem="acquire", scope="sys") != Int32(
                self.world_size
            ):
                nanosleep(150)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("router.load_sizes_and_prefix")
        if self._router_thread_idx == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(load_mbarrier, Int32(matrix_bytes))
            tma_load_1d(
                size_matrix.iterator,
                self._device_workspace.ptr(self.sizes_by_rank_region),
                load_mbarrier,
                Int32(matrix_bytes),
            )

        padded_expert_rounds = ceil_div(self.expert_count_padded, block_thread_count)
        for expert_round in cutlass.range_constexpr(padded_expert_rounds):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if expert < Int32(self.expert_count_padded):
                expert_size = sizes[expert]
                padded_totals[expert] = (
                    (expert_size + Int32(self.token_padding_block - 1))
                    // Int32(self.token_padding_block)
                ) * Int32(self.token_padding_block)
        cute.arch.sync_threads()
        smem_exclusive_prefix(
            padded_totals,
            prefix,
            warp_totals,
            block_thread_count,
            self._router_thread_idx,
            self._router_lane_idx,
            self._router_warp_idx,
        )
        pool_expert_base = self._device_workspace.tensor(self.pool_expert_base_region)
        local_expert_rounds = ceil_div(self.experts_per_rank, block_thread_count)
        for expert_round in cutlass.range_constexpr(local_expert_rounds):
            local_expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if local_expert < Int32(self.experts_per_rank):
                pool_expert_base[local_expert] = (
                    prefix[owner_expert_begin + local_expert] - prefix[owner_expert_begin]
                )

        cute.arch.mbarrier_wait(load_mbarrier, 0)
        iket.range_pop()

        iket.range_push("router.build_push_destinations")
        push_destination_base = self._device_workspace.tensor(self.push_destination_base_region)
        padded_expert_rounds = ceil_div(self.expert_count_padded, block_thread_count)
        for expert_round in cutlass.range_constexpr(padded_expert_rounds):
            global_expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if global_expert < Int32(self.expert_count):
                destination_rank = global_expert // Int32(self.experts_per_rank)
                destination_expert_begin = destination_rank * Int32(self.experts_per_rank)
                destination_pool_base = prefix[global_expert] - prefix[destination_expert_begin]
                local_ring_position = (
                    Int32(self._router_local_rank) - destination_rank + Int32(self.world_size)
                ) % Int32(self.world_size)
                source_ring_offset = Int32(0)
                for ring_position in cutlass.range_constexpr(self.world_size):
                    source_rank = (destination_rank + Int32(ring_position)) % Int32(self.world_size)
                    if Int32(ring_position) < local_ring_position:
                        source_ring_offset = (
                            source_ring_offset + size_matrix[source_rank, global_expert]
                        )
                push_destination_base[global_expert] = destination_pool_base + source_ring_offset
        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            cute.arch.atomic_add(
                self._device_workspace.ptr(self.push_table_ready_region),
                Int32(1),
                sem="release",
                scope="gpu",
            )
        iket.range_pop()

    @cute.jit
    def _load_router_inputs(
        self, topk_indices: cute.Tensor, topk_scores: Optional[cute.Tensor]
    ) -> Tuple[cute.Tensor, Optional[cute.Tensor]]:
        elements_per_vector = 128 // topk_indices.dtype.width
        grid_thread_count = self.router_data_cta_count * self.router_warps_per_cta * 32
        tile_span = elements_per_vector * grid_thread_count
        maximum_elements = self.max_tokens_per_rank * self.topk
        actual_token_count = Int32(self.max_tokens_per_rank)
        actual_elements = Int32(maximum_elements)
        load_round_count = ceil_div(maximum_elements, tile_span)
        elements_per_thread = load_round_count * elements_per_vector

        topk_flat = cute.make_tensor(topk_indices.iterator, cute.make_layout((maximum_elements,)))
        topk_vectors = cute.logical_divide(
            cute.zipped_divide(topk_flat, (tile_span,)), (elements_per_vector, None)
        )
        load_atom = _copy_atom(topk_indices.dtype, 128)
        expert_registers = cute.make_rmem_tensor((elements_per_thread,), cutlass.Int32)
        if cutlass.const_expr(topk_indices.dtype.width == 64):
            raw_indices = cute.make_rmem_tensor((elements_per_thread,), topk_indices.dtype)
            raw_vectors = cute.zipped_divide(raw_indices, (elements_per_vector,))
            for load_round in cutlass.range_constexpr(load_round_count):
                tile_begin = Int32(load_round * tile_span) + self._router_grid_thread_idx * Int32(
                    elements_per_vector
                )
                if tile_begin < actual_elements:
                    cute.copy(
                        load_atom,
                        mark_alignment(
                            topk_vectors[(None, self._router_grid_thread_idx), load_round], 16
                        ),
                        raw_vectors[None, load_round],
                    )
        else:
            expert_vectors = cute.zipped_divide(expert_registers, (elements_per_vector,))
            for load_round in cutlass.range_constexpr(load_round_count):
                tile_begin = Int32(load_round * tile_span) + self._router_grid_thread_idx * Int32(
                    elements_per_vector
                )
                if tile_begin < actual_elements:
                    cute.copy(
                        load_atom,
                        mark_alignment(
                            topk_vectors[(None, self._router_grid_thread_idx), load_round], 16
                        ),
                        expert_vectors[None, load_round],
                    )

        score_registers = None
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_registers = cute.make_rmem_tensor((elements_per_thread,), cutlass.Float32)
            scores_flat = cute.make_tensor(
                topk_scores.iterator, cute.make_layout((maximum_elements,))
            )
            score_vectors = cute.logical_divide(
                cute.zipped_divide(scores_flat, (tile_span,)), (elements_per_vector, None)
            )
            score_atom = _copy_atom(cutlass.Float32, elements_per_vector * 32)
            score_register_vectors = cute.zipped_divide(score_registers, (elements_per_vector,))
            for load_round in cutlass.range_constexpr(load_round_count):
                tile_begin = Int32(load_round * tile_span) + self._router_grid_thread_idx * Int32(
                    elements_per_vector
                )
                if tile_begin < actual_elements:
                    cute.copy(
                        score_atom,
                        mark_alignment(
                            score_vectors[(None, self._router_grid_thread_idx), load_round], 16
                        ),
                        score_register_vectors[None, load_round],
                    )

        expert_registers_u32 = cute.recast_tensor(expert_registers, cutlass.Uint32)
        if cutlass.const_expr(topk_indices.dtype.width == 64):
            raw_indices_i32 = cute.recast_tensor(raw_indices, cutlass.Int32)
        for register_idx in cutlass.range_constexpr(elements_per_thread):
            if cutlass.const_expr(topk_indices.dtype.width == 64):
                expert_registers[register_idx] = raw_indices_i32[2 * register_idx]
            token_idx, _ = self._router_value_coordinate(register_idx, topk_indices.dtype)
            is_invalid = (
                expert_registers_u32[register_idx] >= cutlass.Uint32(self.expert_count)
            ) | (token_idx >= actual_token_count)
            if is_invalid:
                expert_registers[register_idx] = Int32(self.expert_count_padded)
        return expert_registers, score_registers

    @cute.jit
    def _build_histogram(
        self, expert_registers: cute.Tensor, histogram: cute.Tensor
    ) -> cute.Tensor:
        register_count = cute.size(expert_registers)
        within_expert_indices = cute.make_rmem_tensor((register_count,), cutlass.Int32)
        for register_idx in cutlass.range_constexpr(register_count):
            within_expert_indices[register_idx] = Int32(
                cute.arch.atomic_add(
                    histogram.iterator + expert_registers[register_idx],
                    Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
            )
        cute.arch.sync_threads()
        return within_expert_indices

    @cute.jit
    def _sort_router_elements(
        self,
        expert_registers: cute.Tensor,
        within_expert_indices: cute.Tensor,
        score_registers: Optional[cute.Tensor],
        sorted_elements: cute.Tensor,
        expert_run_starts: cute.Tensor,
        topk_index_type: type,
    ) -> None:
        register_count = cute.size(expert_registers)
        for register_idx in cutlass.range_constexpr(register_count):
            token_idx, topk_slot = self._router_value_coordinate(register_idx, topk_index_type)
            flat_topk_index = token_idx * Int32(self.topk) + topk_slot
            destination = (
                expert_run_starts[expert_registers[register_idx]]
                + within_expert_indices[register_idx]
            )
            if cutlass.const_expr(self.apply_topk_at_fc1):
                sorted_elements[destination] = _SortedElement(
                    flat_topk_index, score_registers[register_idx]
                ).pack()
            else:
                sorted_elements[destination] = _SortedElement(flat_topk_index, None).pack()

    @cute.jit
    def _router_value_coordinate(
        self, register_idx: int, topk_index_type: type
    ) -> Tuple[Int32, Int32]:
        elements_per_vector = 128 // topk_index_type.width
        tile_span = (
            elements_per_vector * self.router_data_cta_count * self.router_warps_per_cta * 32
        )
        flat_index = Int32(
            register_idx // elements_per_vector * tile_span + register_idx % elements_per_vector
        ) + self._router_grid_thread_idx * Int32(elements_per_vector)
        return (flat_index // Int32(self.topk), flat_index % Int32(self.topk))

    @cute.jit
    def _dump_contiguous_router_output(
        self, sorted_elements: cute.Tensor, total_valid_routes: Int32
    ) -> None:
        block_thread_count = self.router_warps_per_cta * 32
        metadata_address = self._device_workspace.ptr(self.sorted_metadata_region).toint()
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_address = self._device_workspace.ptr(self.sorted_scores_region).toint()
        dump_round_count = (total_valid_routes + Int32(block_thread_count - 1)) // Int32(
            block_thread_count
        )
        for dump_round in cutlass.range(dump_round_count, unroll=4):
            position = Int32(dump_round * block_thread_count) + self._router_thread_idx
            predicate = Int32(position < total_valid_routes)
            element = _SortedElement.from_packed(sorted_elements[position])
            metadata = TokenSrcMetadata(
                src_rank=Int32(self._router_local_rank),
                src_token=(element.flat_topk_index // Int32(self.topk)),
                src_topk=(element.flat_topk_index % Int32(self.topk)),
            )
            stg_b64(
                metadata_address + Int64(position) * Int64(TokenSrcMetadata.nbytes),
                metadata.pack(),
                predicate,
            )
            if cutlass.const_expr(self.apply_topk_at_fc1):
                stg_f32(score_address + Int64(position) * Int64(4), element.topk_score, predicate)

    @cute.jit
    def _dump_router_output_by_expert(
        self,
        histogram: cute.Tensor,
        expert_run_starts: cute.Tensor,
        expert_dump_bases: cute.Tensor,
        sorted_elements: cute.Tensor,
    ) -> None:
        metadata_address = self._device_workspace.ptr(self.sorted_metadata_region).toint()
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_address = self._device_workspace.ptr(self.sorted_scores_region).toint()
        expert_round_count = ceil_div(self.expert_count_padded, self.router_warps_per_cta)
        for expert_round in cutlass.range_constexpr(expert_round_count):
            expert = self._router_warp_idx + Int32(expert_round * self.router_warps_per_cta)
            if expert < Int32(self.expert_count_padded):
                run_begin = expert_run_starts[expert]
                run_length = histogram[expert]
                dump_begin = expert_dump_bases[expert]
                route_round_count = (run_length + Int32(31)) // Int32(32)
                for route_round in cutlass.range(route_round_count, unroll=1):
                    route = Int32(route_round) * Int32(32) + self._router_lane_idx
                    predicate = Int32(route < run_length)
                    element = _SortedElement.from_packed(
                        sorted_elements[predicate * (run_begin + route)]
                    )
                    output_position = dump_begin + route
                    metadata = TokenSrcMetadata(
                        src_rank=Int32(self._router_local_rank),
                        src_token=(element.flat_topk_index // Int32(self.topk)),
                        src_topk=(element.flat_topk_index % Int32(self.topk)),
                    )
                    stg_b64(
                        metadata_address + Int64(output_position) * Int64(TokenSrcMetadata.nbytes),
                        metadata.pack(),
                        predicate,
                    )
                    if cutlass.const_expr(self.apply_topk_at_fc1):
                        stg_f32(
                            score_address + Int64(output_position) * Int64(4),
                            element.topk_score,
                            predicate,
                        )

    @cute.jit
    def local_expert_sizes(
        self, device_workspace: DeviceWorkspace, local_rank: Int32
    ) -> cute.Tensor:
        """Return this rank's contiguous expert-size view."""
        sizes = device_workspace.tensor(self.sizes_region)
        expert_begin = local_rank * Int32(self.experts_per_rank)
        return cute.make_tensor(
            sizes.iterator + expert_begin, cute.make_layout((self.experts_per_rank,))
        )

    @property
    def metadata_ready_target(self) -> int:
        return self.router_push_cta_count * self.world_size

    @cute.jit
    def sizes_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.sizes_region)

    @cute.jit
    def pool_expert_base_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.pool_expert_base_region)

    @cute.jit
    def token_src_metadata_pointer(self, device_workspace: DeviceWorkspace) -> cute.Pointer:
        return device_workspace.ptr(self.token_src_metadata_region)

    @cute.jit
    def wait_for_sizes_ready(
        self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        lane_idx = thread_idx % Int32(32)
        if lane_idx == Int32(0):
            sizes_ready = device_workspace.ptr(self.sizes_ready_region)
            while cute.arch.load(sizes_ready, Int32, sem="acquire", scope="sys") != Int32(
                self.world_size
            ):
                nanosleep(sleep_cycles)
        cute.arch.sync_warp()

    @cute.jit
    def wait_for_metadata_ready(
        self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        lane_idx = thread_idx % Int32(32)
        if lane_idx == Int32(0):
            metadata_ready = device_workspace.ptr(self.metadata_ready_region)
            while cute.arch.load(metadata_ready, Int32, sem="acquire", scope="sys") != Int32(
                self.metadata_ready_target
            ):
                nanosleep(sleep_cycles)
        cute.arch.sync_warp()

    @cute.jit
    def token_src_metadata_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.token_src_metadata_region)

    @cute.jit
    def fc1_topk_scores_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.apply_topk_at_fc1):
            return None
        return device_workspace.tensor(self.fc1_topk_scores_region)


class TokenCommNonDeterministic(KernelComponent):
    """Public fused-kernel token communication component."""

    transfer_warp_count: ClassVar[int] = 4
    transfer_thread_count: ClassVar[int] = transfer_warp_count * 32
    standalone_chunk_bytes: ClassVar[int] = 2048
    minimum_pacing_window_cycles: ClassVar[int] = 512
    standalone_max_backoff_cycles: ClassVar[int] = 500
    adaptive_minimum_sleep_cycles: ClassVar[int] = 50
    transfer_lifetime_barrier_id: ClassVar[int] = 9
    grid_sync_barrier_id: ClassVar[int] = 10
    standalone_size_barrier_id: ClassVar[int] = 11
    token_in_size_barrier_id: ClassVar[int] = 12

    fc1_ready_region = "nvlink.token_comm.fc1_ready"
    fc1_activation_region = "nvlink.token_comm.fc1_activation"
    fc1_activation_sf_region = "nvlink.token_comm.fc1_activation_sf"
    fc2_done_region = "nvlink.token_comm.fc2_done"
    fc2_activation_region = "nvlink.token_comm.fc2_activation"
    fc2_activation_sf_region = "nvlink.token_comm.fc2_activation_sf"
    pre_reduced_activation_region = "nvlink.token_comm.pre_reduced_activation"
    pre_reduced_activation_sf_region = "nvlink.token_comm.pre_reduced_activation_sf"
    token_back_schedule_region = "nvlink.token_comm.token_back_schedule"

    token_in_mbarrier_region = "nvlink.token_comm.main_smem.token_in_mbarriers"
    token_back_mbarrier_region = "nvlink.token_comm.main_smem.token_back_mbarriers"
    expert_sizes_smem_region = "nvlink.token_comm.main_smem.expert_sizes"
    token_in_activation_smem_region = "nvlink.token_comm.main_smem.token_in_activation"
    token_in_sf_smem_region = "nvlink.token_comm.main_smem.token_in_sf"
    token_back_activation_smem_region = "nvlink.token_comm.main_smem.token_back_activation"
    token_back_sf_smem_region = "nvlink.token_comm.main_smem.token_back_sf"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "world_size": int,
            "expert_count": int,
            "topk": int,
            "max_tokens_per_rank": int,
            "hidden_size": int,
            "quant_kind": str,
            "combine_format": CombineFormat,
            "apply_topk_at_fc1": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "token_padding_block": int,
            "sf_padding_block": int,
            "tokens_per_fc1_ready_slot": int,
            "fc2_done_signals_per_token_tile": int,
            "promised_launchable_sm_count": int,
            "token_in_flag_batch": int,
            "token_back_mode": str,
            "token_back_schedule_mode": str,
            "reduce_topk_in_kernel": bool,
            "router_smem_limit_bytes": OptionalRequirement(int),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.world_size = problem_desc["world_size"]
        self.expert_count = problem_desc["expert_count"]
        self.topk = problem_desc["topk"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.combine_format = problem_desc["combine_format"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.tokens_per_fc1_ready_slot = impl_desc["tokens_per_fc1_ready_slot"]
        self.fc2_done_signals_per_token_tile = impl_desc["fc2_done_signals_per_token_tile"]
        self.promised_launchable_sm_count = impl_desc["promised_launchable_sm_count"]
        self.token_in_flag_batch = impl_desc["token_in_flag_batch"]
        self.token_back_mode: TokenBackMode = impl_desc["token_back_mode"]
        self.token_back_schedule_mode: TokenBackScheduleMode = impl_desc["token_back_schedule_mode"]
        self.reduce_topk_in_kernel = impl_desc["reduce_topk_in_kernel"]
        self.router_smem_limit_bytes = impl_desc.get("router_smem_limit_bytes", 227 * 1024)

        self._validate_configuration()
        self._router = _MetadataPushRouter(problem_desc, impl_desc)
        self._nvlink_barrier = NvlinkBarrier(
            world_size=self.world_size, barrier_id=self.grid_sync_barrier_id
        )
        self._device_workspace = None
        self._token_comm_args = None
        self._local_rank = None
        self._linear_cta_idx = None
        self._transfer_warp_idx = None
        self._lane_idx = None

    def _validate_configuration(self) -> None:
        positive_fields = (
            "world_size",
            "expert_count",
            "topk",
            "max_tokens_per_rank",
            "hidden_size",
            "token_padding_block",
            "sf_padding_block",
            "tokens_per_fc1_ready_slot",
            "promised_launchable_sm_count",
            "router_smem_limit_bytes",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")
        if self.expert_count % self.world_size != 0:
            raise ValueError(
                f"expert_count must be divisible by world_size, got {self.expert_count} and {self.world_size}."
            )
        if self.expert_count > 16384:
            raise NotImplementedError("TokenComm supports at most 16384 global experts.")
        if self.topk > self.expert_count:
            raise ValueError(
                f"topk must not exceed expert_count, got {self.topk} and {self.expert_count}."
            )
        if self.token_back_mode not in ("epi_warps", "standalone_warps", "reuse_dispatch_warps"):
            raise ValueError(f"Unsupported token_back_mode {self.token_back_mode!r}.")
        if self.token_back_schedule_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"token_back_schedule_mode must be static or atomic_counter, got {self.token_back_schedule_mode!r}."
            )
        if not 1 <= self.token_in_flag_batch <= 32:
            raise ValueError(
                f"token_in_flag_batch must be in [1, 32], got {self.token_in_flag_batch}."
            )
        if self.tokens_per_fc1_ready_slot % self.token_padding_block != 0:
            raise ValueError("tokens_per_fc1_ready_slot must be divisible by token_padding_block.")
        if self.token_back_enabled and self.fc2_done_signals_per_token_tile <= 0:
            raise ValueError(
                "fc2_done_signals_per_token_tile must be positive when token-back is enabled."
            )
        if self.reduce_topk_in_kernel and self.combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError("In-kernel top-k reduction requires BF16 combine data.")
        element_block = self.activation_sf_vector_size * 4
        if self.hidden_size % element_block != 0:
            raise ValueError(
                f"{self.quant_kind} requires hidden_size divisible by {element_block}."
            )
        if self.sf_padding_block % 128 != 0:
            raise ValueError("sf_padding_block must be a multiple of 128.")

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.world_size

    @property
    def activation_dtype(self) -> type:
        return self.quant_kind.activation_dtype

    @property
    def activation_sf_dtype(self) -> type:
        return self.quant_kind.sf_dtype

    @property
    def activation_sf_vector_size(self) -> int:
        return self.quant_kind.sf_vec_size

    @property
    def bytes_per_token(self) -> int:
        return self.hidden_size * int(self.activation_dtype.width) // 8

    @property
    def activation_sf_hidden_padded(self) -> int:
        valid_hidden = self.hidden_size // self.activation_sf_vector_size
        elements_per_16_bytes = 128 // int(self.activation_sf_dtype.width)
        return int(round_up(valid_hidden, elements_per_16_bytes))

    @property
    def combine_sf_hidden_padded(self) -> int:
        if not self.combine_format.is_quantized:
            return 0
        valid_hidden = self.hidden_size // self.combine_format.scale_block
        elements_per_16_bytes = 128 // int(self.combine_format.scale_dtype.width)
        return int(round_up(valid_hidden, elements_per_16_bytes))

    @property
    def token_back_push_data(self) -> bool:
        return self.token_back_mode != "epi_warps"

    @property
    def token_back_push_sf(self) -> bool:
        return self.combine_format.is_quantized

    @property
    def token_back_enabled(self) -> bool:
        return self.token_back_push_data or self.token_back_push_sf

    @property
    def worst_case_token_count(self) -> int:
        return self._router.worst_case_token_count

    @property
    def worst_case_sf_token_count(self) -> int:
        return self._router.worst_case_padded_tokens(self.sf_padding_block)

    @property
    def max_fc1_ready_slot_count(self) -> int:
        return (
            self._router.worst_case_padded_tokens(self.tokens_per_fc1_ready_slot)
            // self.tokens_per_fc1_ready_slot
        )

    @property
    def router_smem_workspace(self) -> SmemWorkspace:
        return self._router.router_smem_workspace

    @property
    def expert_count_padded(self) -> int:
        return self._router.expert_count_padded

    @property
    def expert_count_with_trash(self) -> int:
        return self._router.expert_count_with_trash

    @property
    def router_elements_per_lane(self) -> int:
        return self._router.router_elements_per_lane

    @property
    def router_warps_per_cta(self) -> int:
        return self._router.router_warps_per_cta

    @property
    def router_data_cta_count(self) -> int:
        return self._router.router_data_cta_count

    @property
    def router_tokens_per_cta(self) -> int:
        return self._router.router_tokens_per_cta

    @property
    def router_push_cta_count(self) -> int:
        return self._router.router_push_cta_count

    @property
    def router_grid_cta_count(self) -> int:
        return self._router.router_grid_cta_count

    @property
    def metadata_ready_target(self) -> int:
        return self._router.metadata_ready_target

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        self._router.register_device_workspace(workspace)
        self._register_main_workspace(workspace)
        self._nvlink_barrier.register_device_workspace(workspace)

    @cute.jit
    def launch_router(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper_host,
        device_workspace: DeviceWorkspace,
        stream: cuda.CUstream,
    ) -> None:
        self._router.launch_router(
            topk_indices,
            topk_scores,
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper_host,
            device_workspace,
            stream,
        )

    @cute.jit
    def local_expert_sizes(
        self, device_workspace: DeviceWorkspace, local_rank: Int32
    ) -> cute.Tensor:
        return self._router.local_expert_sizes(device_workspace, local_rank)

    @cute.jit
    def wait_for_sizes_ready(
        self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000
    ) -> None:
        self._router.wait_for_sizes_ready(device_workspace, sleep_cycles)

    @cute.jit
    def token_src_metadata_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return self._router.token_src_metadata_tensor(device_workspace)

    @cute.jit
    def fc1_topk_scores_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        return self._router.fc1_topk_scores_tensor(device_workspace)

    @cute.jit
    def assign_device_members(
        self,
        *,
        device_workspace: DeviceWorkspace,
        token_comm_args: TokenCommArgs,
        local_rank: Int32,
        linear_cta_idx: Int32,
    ) -> None:
        self._device_workspace = device_workspace
        self._token_comm_args = token_comm_args
        self._local_rank = local_rank
        self._linear_cta_idx = linear_cta_idx
        thread_idx, _, _ = cute.arch.thread_idx()
        transfer_thread_idx = thread_idx % Int32(self.transfer_thread_count)
        self._transfer_warp_idx = cute.arch.make_warp_uniform(transfer_thread_idx // Int32(32))
        self._lane_idx = transfer_thread_idx % Int32(32)
        self._nvlink_barrier.assign_device_members(
            device_workspace, token_comm_args.peer_rank_ptr_mapper
        )

    def remove_device_members(self) -> None:
        self._nvlink_barrier.remove_device_members()
        self._device_workspace = None
        self._token_comm_args = None
        self._local_rank = None
        self._linear_cta_idx = None
        self._transfer_warp_idx = None
        self._lane_idx = None

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "TokenCommNonDeterministic":
        if values:
            raise ValueError("TokenCommNonDeterministic carries no MLIR values.")
        return self

    def _register_main_workspace(self, workspace: DeviceWorkspace) -> None:
        """Register the fused kernel's GMEM regions. Three groups, each stating its own existence condition.

        FC1 pools, always present: the dispatched payload ``token_in`` pulls in, addressed in POOL index space
        (per-expert padded runs of tokens routed into this rank's experts, from every rank).

        Token-back machinery, iff ``token_back_enabled``: the transfer warps' counters plus the rank-local FC2
        staging they read, also in pool index space. ``epi_warps`` needs none of it, because there the FC2
        epilogue reaches the peers itself. The scale plane is registered separately from the data plane and is
        staged locally in EVERY mode: pushing scales per token would scatter one warp's 32 lanes across up to 32
        ranks and explode the NVLink request count.

        Combine plane, iff ``not reduce_topk_in_kernel``: the symmetric per-topk landing zone peers deliver into,
        addressed in (source token, source topk) space. It is the DESTINATION of the round trip whose source is
        the staging above, so its condition is deliberately independent of who performs the transfer --
        ``epi_warps`` with a separate reduce registers this plane while registering no token-back machinery at
        all. Peers reach it through the symmetric heap, so it cannot be a caller tensor: only this component
        knows the wire dtype and the padded scale stride. Its ``data`` reset keeps it out of both the host zero
        prefix and the per-launch tail reset, since every cell it exposes is rewritten each launch and the plane
        is far too large to be worth clearing.
        """
        workspace.register(
            self.fc1_ready_region,
            cutlass.Int32,
            (self.max_fc1_ready_slot_count,),
            buffer_space="local",
            reset="tail_reset",
        )
        activation_element_count = self.worst_case_token_count * self.hidden_size
        workspace.register(
            self.fc1_activation_region,
            self.activation_dtype,
            (activation_element_count,),
            buffer_space="local",
            byte_alignment=128,
        )
        activation_sf_element_count = (
            self.worst_case_sf_token_count * self.activation_sf_hidden_padded
        )
        workspace.register(
            self.fc1_activation_sf_region,
            self.activation_sf_dtype,
            (activation_sf_element_count,),
            buffer_space="local",
            byte_alignment=128,
        )

        if self.token_back_enabled:
            workspace.register(
                self.fc2_done_region,
                cutlass.Int32,
                (self.experts_per_rank,),
                buffer_space="local",
                reset="tail_reset",
            )
        if self.token_back_push_data:
            fc2_element_count = self.worst_case_token_count * self.hidden_size
            workspace.register(
                self.fc2_activation_region,
                self.combine_format.act_dtype,
                (fc2_element_count,),
                buffer_space="local",
                byte_alignment=128,
            )
        if self.token_back_push_sf:
            fc2_sf_element_count = self.worst_case_token_count * self.combine_sf_hidden_padded
            workspace.register(
                self.fc2_activation_sf_region,
                self.combine_format.scale_dtype,
                (fc2_sf_element_count,),
                buffer_space="local",
                byte_alignment=128,
            )
        if self.token_back_enabled and self.token_back_schedule_mode == "atomic_counter":
            workspace.register(
                self.token_back_schedule_region,
                cutlass.Int32,
                (1,),
                buffer_space="local",
                reset="tail_reset",
            )

        if not self.reduce_topk_in_kernel:
            workspace.register(
                self.pre_reduced_activation_region,
                self.combine_format.act_dtype,
                (self.max_tokens_per_rank, self.topk, self.hidden_size),
                buffer_space="shared",
                mem_order=(2, 1, 0),
                byte_alignment=128,
            )
            if self.combine_format.is_quantized:
                workspace.register(
                    self.pre_reduced_activation_sf_region,
                    self.combine_format.scale_dtype,
                    (self.max_tokens_per_rank, self.topk, self.combine_sf_hidden_padded),
                    buffer_space="shared",
                    mem_order=(2, 1, 0),
                    byte_alignment=128,
                )

    def register_smem_regions(self, workspace: SmemWorkspace) -> None:
        workspace.register_mbarrier(self.token_in_mbarrier_region, self.transfer_warp_count)
        if self.token_back_enabled:
            workspace.register_mbarrier(self.token_back_mbarrier_region, self.transfer_warp_count)
        workspace.register_tensor(
            self.expert_sizes_smem_region,
            cutlass.Int32,
            (self.experts_per_rank,),
            byte_alignment=16,
        )
        transfer_overlay = workspace.create_overlay("nvlink.token_comm.main_smem.transfer")
        token_in_lifetime = transfer_overlay.add_lifetime("token_in")
        token_in_lifetime.register_tensor(
            self.token_in_activation_smem_region,
            self.activation_dtype,
            (self.transfer_warp_count, self.hidden_size),
            byte_alignment=16,
        )
        token_in_lifetime.register_tensor(
            self.token_in_sf_smem_region,
            self.activation_sf_dtype,
            (
                self.transfer_warp_count,
                (self.activation_sf_vector_size, self.activation_sf_hidden_padded),
            ),
            stride=(self.activation_sf_hidden_padded, (0, 1)),
            byte_alignment=16,
        )
        if not self.token_back_enabled:
            return

        if self.token_back_mode == "standalone_warps":
            token_back_lifetime = workspace.create_overlay(
                "nvlink.token_comm.main_smem.standalone_token_back"
            ).add_lifetime("token_back")
        else:
            token_back_lifetime = transfer_overlay.add_lifetime("token_back")

        if self.token_back_mode == "standalone_warps":
            available_bytes_per_warp = self.standalone_chunk_bytes
        else:
            activation_bytes = self.bytes_per_token
            sf_bytes = self.activation_sf_hidden_padded * int(self.activation_sf_dtype.width) // 8
            available_bytes_per_warp = activation_bytes + sf_bytes

        if self.token_back_push_data:
            bytes_per_output_token = (
                self.hidden_size * int(self.combine_format.act_dtype.width) // 8
            )
            if self.token_back_mode == "standalone_warps":
                chunk_bytes = self.standalone_chunk_bytes
            elif available_bytes_per_warp < bytes_per_output_token:
                chunk_bytes = self.bytes_per_token
            else:
                chunk_bytes = bytes_per_output_token
            if (
                self.token_back_mode != "standalone_warps"
                and bytes_per_output_token % chunk_bytes != 0
            ):
                raise ValueError("Token-back data chunk bytes must divide one row.")
            chunk_elements = chunk_bytes * 8 // int(self.combine_format.act_dtype.width)
            token_back_lifetime.register_tensor(
                self.token_back_activation_smem_region,
                self.combine_format.act_dtype,
                (self.transfer_warp_count, chunk_elements),
                byte_alignment=16,
            )
        if self.token_back_push_sf:
            sf_row_bytes = (
                self.combine_sf_hidden_padded * int(self.combine_format.scale_dtype.width) // 8
            )
            if sf_row_bytes > available_bytes_per_warp:
                raise ValueError("Token-back scale row exceeds its per-warp stage.")
            token_back_lifetime.register_tensor(
                self.token_back_sf_smem_region,
                self.combine_format.scale_dtype,
                (
                    self.transfer_warp_count,
                    (self.combine_format.scale_block, self.combine_sf_hidden_padded),
                ),
                stride=(self.combine_sf_hidden_padded, (0, 1)),
                byte_alignment=16,
            )

    @cute.jit
    def fc1_ready_counter_pointer(self, device_workspace: DeviceWorkspace) -> cute.Pointer:
        return device_workspace.ptr(self.fc1_ready_region)

    @cute.jit
    def fc1_activation_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return cute.make_tensor(
            device_workspace.ptr(self.fc1_activation_region),
            cute.make_layout(
                (self.worst_case_token_count, self.hidden_size), stride=(self.hidden_size, 1)
            ),
        )

    @cute.jit
    def fc1_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        layout = tile_atom_to_shape_SF(
            (self.worst_case_sf_token_count, self.hidden_size, 1), self.activation_sf_vector_size
        )
        return cute.make_tensor(
            device_workspace.ptr(self.fc1_activation_sf_region), cute.select(layout, mode=[0, 1])
        )

    @cute.jit
    def fc2_done_counter_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.token_back_enabled):
            return None
        return device_workspace.tensor(self.fc2_done_region)

    @cute.jit
    def fc2_activation_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.token_back_push_data):
            return None
        return cute.make_tensor(
            device_workspace.ptr(self.fc2_activation_region),
            cute.make_layout(
                (self.worst_case_token_count, 1, self.hidden_size),
                stride=(self.hidden_size, self.hidden_size, 1),
            ),
        )

    @cute.jit
    def fc2_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.token_back_push_sf):
            return None
        return cute.make_tensor(
            device_workspace.ptr(self.fc2_activation_sf_region),
            cute.make_layout(
                (
                    self.worst_case_token_count,
                    1,
                    (
                        self.combine_format.scale_block,
                        self.hidden_size // self.combine_format.scale_block,
                    ),
                ),
                stride=(self.combine_sf_hidden_padded, self.combine_sf_hidden_padded, (0, 1)),
            ),
        )

    @cute.jit
    def pre_reduced_activation_tensor(
        self, device_workspace: DeviceWorkspace
    ) -> Optional[cute.Tensor]:
        """The (tokens, topk, hidden) combine staging plane, or None under in-kernel top-k reduction."""
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            return None
        return device_workspace.tensor(self.pre_reduced_activation_region)

    @cute.jit
    def pre_reduced_activation_sf_tensor(
        self, device_workspace: DeviceWorkspace
    ) -> Optional[cute.Tensor]:
        """The scale plane parallel to ``pre_reduced_activation_tensor``; only quantized wire formats carry one."""
        if cutlass.const_expr(self.reduce_topk_in_kernel or not self.combine_format.is_quantized):
            return None
        return device_workspace.tensor(self.pre_reduced_activation_sf_region)

    @cute.jit
    def token_in(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """Wait for pushed metadata, then pull activation payloads into local pools."""
        transfer_warp_idx = self._transfer_warp_idx
        lane_idx = self._lane_idx
        global_warp_idx = self._linear_cta_idx * Int32(self.transfer_warp_count) + transfer_warp_idx
        global_warp_count = Int32(self.promised_launchable_sm_count * self.transfer_warp_count)

        sizes = self._router.sizes_tensor(self._device_workspace)
        pool_expert_bases = self._router.pool_expert_base_tensor(self._device_workspace)
        token_metadata_pointer = self._router.token_src_metadata_pointer(self._device_workspace)

        iket.range_push("token_in.wait_sizes_ready")
        self._router.wait_for_sizes_ready(self._device_workspace)
        iket.range_pop()
        iket.range_push("token_in.stage_sizes")
        owned_sizes = smem_workspace.tensor(self.expert_sizes_smem_region, smem_base)
        owner_expert_begin = self._local_rank * Int32(self.experts_per_rank)
        source_sizes = cute.make_tensor(
            sizes.iterator + owner_expert_begin, cute.make_layout((self.experts_per_rank,))
        )
        copy_elements = 4 if self.experts_per_rank % 4 == 0 else 1
        source_size_vectors = cute.zipped_divide(
            (
                mark_alignment(source_sizes, 16)
                if cutlass.const_expr(copy_elements == 4)
                else source_sizes
            ),
            (copy_elements,),
        )
        destination_size_vectors = cute.zipped_divide(owned_sizes, (copy_elements,))
        size_vector_count = cute.size(destination_size_vectors, mode=[1])
        size_copy_atom = _copy_atom(cutlass.Int32, copy_elements * 32)
        size_copy_rounds = ceil_div(size_vector_count, self.transfer_thread_count)
        size_copy_registers = cute.make_rmem_tensor(
            (copy_elements, size_copy_rounds), cutlass.Int32
        )
        transfer_thread_idx = transfer_warp_idx * Int32(32) + lane_idx
        for size_copy_round in cutlass.range_constexpr(size_copy_rounds):
            vector_idx = Int32(size_copy_round * self.transfer_thread_count) + transfer_thread_idx
            if vector_idx < Int32(size_vector_count):
                cute.copy(
                    size_copy_atom,
                    source_size_vectors[None, vector_idx],
                    size_copy_registers[None, size_copy_round],
                )
        iket.range_pop()

        iket.range_push("token_in.wait_metadata_ready")
        self._router.wait_for_metadata_ready(self._device_workspace)
        iket.range_pop()

        for size_copy_round in cutlass.range_constexpr(size_copy_rounds):
            vector_idx = Int32(size_copy_round * self.transfer_thread_count) + transfer_thread_idx
            if vector_idx < Int32(size_vector_count):
                cute.copy(
                    size_copy_atom,
                    size_copy_registers[None, size_copy_round],
                    destination_size_vectors[None, vector_idx],
                )
        iket.range_push("token_in.size_barrier")
        token_in_size_barrier = pipeline.NamedBarrier(
            barrier_id=self.token_in_size_barrier_id, num_threads=self.transfer_thread_count
        )
        token_in_size_barrier.arrive_and_wait()
        iket.range_pop()
        if cutlass.const_expr(self.token_back_mode == "standalone_warps"):
            sizes_ready_barrier = pipeline.NamedBarrier(
                barrier_id=self.standalone_size_barrier_id,
                num_threads=2 * self.transfer_thread_count,
            )
            sizes_ready_barrier.arrive()

        iket.range_push("token_in.pull_payload")
        token_in_mbarriers = smem_workspace.ptr(self.token_in_mbarrier_region, smem_base)
        token_in_activation = smem_workspace.tensor(self.token_in_activation_smem_region, smem_base)
        token_in_sf = smem_workspace.tensor(self.token_in_sf_smem_region, smem_base)
        warp_mbarrier = token_in_mbarriers + transfer_warp_idx
        warp_activation_stage = token_in_activation[transfer_warp_idx, None]
        warp_sf_stage = token_in_sf[transfer_warp_idx, (None, None)]
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(warp_mbarrier, 1)
        cute.arch.sync_warp()

        fc1_activation_pointer = self._device_workspace.ptr(self.fc1_activation_region)
        fc1_activation_sf = self.fc1_activation_sf_tensor(self._device_workspace)
        fc1_ready_counter = self._device_workspace.ptr(self.fc1_ready_region)
        activation_bytes = (
            cute.cosize(warp_activation_stage) * int(self.activation_dtype.width) // 8
        )
        activation_sf_bytes = cute.cosize(warp_sf_stage) * int(self.activation_sf_dtype.width) // 8
        sf_copy_elements = 4
        source_sf_values = cute.slice_(warp_sf_stage, (0, None))
        source_sf_vectors = cute.zipped_divide(source_sf_values, (sf_copy_elements,))
        sf_copy_atom = _copy_atom(
            self.activation_sf_dtype, sf_copy_elements * int(self.activation_sf_dtype.width)
        )

        next_dense_token = global_warp_idx
        expert_valid_begin = Int32(0)
        expert_sf_begin = Int32(0)
        expert_ready_slot_begin = Int32(0)
        pull_phase = Int32(0)
        flag_tracker = make_flag_batch_tracker(
            use_async=self.token_in_flag_batch == 1,
            flag_address=Int64(0),
            accumulated_flags=Int32(0),
            phase=Int32(0),
            thread_idx=lane_idx,
        )

        local_expert = Int32(0)
        while local_expert < Int32(self.experts_per_rank):
            expert_token_count = owned_sizes[local_expert]
            expert_valid_end = expert_valid_begin + expert_token_count
            pull_count = Int32(0)
            if next_dense_token < expert_valid_end:
                pull_count = (
                    expert_valid_end - next_dense_token + global_warp_count - Int32(1)
                ) // global_warp_count

            for pull_round in cutlass.range(pull_count, unroll=1):
                dense_token_idx = next_dense_token + Int32(pull_round) * global_warp_count
                token_in_expert = dense_token_idx - expert_valid_begin
                pool_token_idx = pool_expert_bases[local_expert] + token_in_expert
                sf_token_idx = expert_sf_begin + token_in_expert
                ready_slot_idx = expert_ready_slot_begin + token_in_expert // Int32(
                    self.tokens_per_fc1_ready_slot
                )

                metadata = TokenSrcMetadata.load(
                    token_metadata_pointer.toint()
                    + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                )
                peer_offset = self._token_comm_args.peer_rank_ptr_mapper.map(
                    Int64(0), metadata.src_rank, Int64(0)
                )
                remote_activation_address = (
                    self._token_comm_args.activation.iterator.toint()
                    + peer_offset
                    + Int64(metadata.src_token) * Int64(self.bytes_per_token)
                )
                remote_sf = cute.make_tensor(
                    cute.make_ptr(
                        self._token_comm_args.activation_sf.dtype,
                        self._token_comm_args.activation_sf.iterator.toint() + peer_offset,
                        AddressSpace.gmem,
                        assumed_align=(self._token_comm_args.activation_sf.iterator.max_alignment),
                    ),
                    self._token_comm_args.activation_sf.layout,
                )
                remote_sf_row = remote_sf[Int64(metadata.src_token), None]

                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        warp_mbarrier, Int32(activation_bytes + activation_sf_bytes)
                    )
                    tma_load_1d(
                        warp_activation_stage.iterator,
                        Int64(remote_activation_address),
                        warp_mbarrier,
                        Int32(activation_bytes),
                    )
                    tma_load_1d(
                        warp_sf_stage.iterator,
                        remote_sf_row.iterator,
                        warp_mbarrier,
                        Int32(activation_sf_bytes),
                    )
                cute.arch.sync_warp()
                cute.arch.mbarrier_wait(warp_mbarrier, pull_phase)

                destination_activation = cute.make_ptr(
                    self.activation_dtype,
                    fc1_activation_pointer.toint()
                    + Int64(pool_token_idx) * Int64(self.bytes_per_token),
                    AddressSpace.gmem,
                    assumed_align=16,
                )
                with cute.arch.elect_one():
                    cp_async_bulk_s2g(
                        destination_activation,
                        warp_activation_stage.iterator,
                        Int32(activation_bytes),
                    )
                cute.arch.sync_warp()
                cute.arch.cp_async_bulk_commit_group()

                destination_sf_row = fc1_activation_sf[Int64(sf_token_idx), ((None, None), None)]
                destination_sf_values = cute.slice_(destination_sf_row, (0, None, None))
                destination_sf_values = cute.group_modes(destination_sf_values, 0, 2)
                destination_sf_vectors = cute.zipped_divide(
                    destination_sf_values, (sf_copy_elements,)
                )
                sf_vector_count = cute.size(destination_sf_vectors, mode=[1])
                for sf_round in cutlass.range_constexpr(ceil_div(sf_vector_count, 32)):
                    sf_vector_idx = Int32(sf_round * 32) + lane_idx
                    if sf_vector_idx < Int32(sf_vector_count):
                        cute.copy(
                            sf_copy_atom,
                            source_sf_vectors[None, sf_vector_idx],
                            destination_sf_vectors[None, sf_vector_idx],
                        )

                cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.sync_warp()
                ready_address = (fc1_ready_counter + ready_slot_idx).toint()
                flag_tracker = flag_tracker.accumulate(
                    Int32(0), self.token_in_flag_batch, ready_address
                )
                cute.arch.sync_warp()
                pull_phase = pull_phase ^ Int32(1)

            next_dense_token = next_dense_token + pull_count * global_warp_count
            expert_valid_begin = expert_valid_end
            expert_sf_begin = expert_sf_begin + (
                (expert_token_count + Int32(self.sf_padding_block - 1))
                // Int32(self.sf_padding_block)
            ) * Int32(self.sf_padding_block)
            expert_ready_slot_begin = expert_ready_slot_begin + (
                (expert_token_count + Int32(self.tokens_per_fc1_ready_slot - 1))
                // Int32(self.tokens_per_fc1_ready_slot)
            )
            local_expert = local_expert + Int32(1)

        flag_tracker.fire()
        cute.arch.sync_warp()
        iket.range_pop()
        if cutlass.const_expr(
            self.token_back_enabled and self.token_back_mode != "standalone_warps"
        ):
            iket.range_push("token_in.transfer_barrier")
            transfer_lifetime_barrier = pipeline.NamedBarrier(
                barrier_id=self.transfer_lifetime_barrier_id, num_threads=self.transfer_thread_count
            )
            transfer_lifetime_barrier.arrive_and_wait()
            iket.range_pop()

    @cute.jit
    def _stateless_pace(self, reference_window: Int32, current_window: Int32) -> None:
        sleep_cycles = Int32(0)
        if current_window < reference_window:
            sleep_cycles = reference_window - current_window
        elif current_window > reference_window:
            sleep_cycles = cutlass.min(
                current_window - reference_window, Int32(self.standalone_max_backoff_cycles)
            )
        if sleep_cycles > Int32(0):
            nanosleep(sleep_cycles)

    @cute.jit
    def _adaptive_pace(
        self, average_window: Int32, current_window: Int32, low_window: int, high_window: int
    ) -> Int32:
        sleep_cycles = Int32(0)
        if current_window > average_window:
            sleep_cycles = current_window - average_window
            average_window = average_window + (
                (current_window - average_window + Int32(3)) // Int32(4)
            )
            if sleep_cycles > Int32(high_window):
                sleep_cycles = Int32(high_window)
        else:
            sleep_cycles = average_window - current_window
            average_window = average_window - (
                (average_window - current_window + Int32(3)) // Int32(4)
            )
        # with cute.arch.elect_one():
        # cute.printf("avg_window: {}, current_window: {}, sleep_cycles: {}", average_window, current_window,
        # sleep_cycles)
        if sleep_cycles > Int32(self.adaptive_minimum_sleep_cycles):
            nanosleep(sleep_cycles)
        if average_window > Int32(high_window):
            average_window = Int32(high_window)
        if average_window < Int32(low_window):
            average_window = Int32(low_window)
        return average_window

    @cute.jit
    def token_back(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """Push completed FC2 data and scale rows to source ranks."""
        transfer_warp_idx = self._transfer_warp_idx
        lane_idx = self._lane_idx
        if cutlass.const_expr(not self.token_back_enabled):
            return
        if cutlass.const_expr(
            self.combine_format.is_quantized
            and self._token_comm_args.pre_reduced_activation_sf is None
        ):
            raise ValueError("Quantized token-back requires a scale destination.")

        global_worker_idx = (
            self._linear_cta_idx * Int32(self.transfer_warp_count) + transfer_warp_idx
        )
        global_worker_count = Int32(self.promised_launchable_sm_count * self.transfer_warp_count)
        token_back_mbarriers = smem_workspace.ptr(self.token_back_mbarrier_region, smem_base)
        worker_mbarrier = token_back_mbarriers + transfer_warp_idx
        if cutlass.const_expr(self.token_back_push_data):
            token_back_activation = smem_workspace.tensor(
                self.token_back_activation_smem_region, smem_base
            )
            worker_activation_stage = token_back_activation[transfer_warp_idx, None]
            activation_chunk_bytes = (
                cute.cosize(worker_activation_stage) * int(self.combine_format.act_dtype.width) // 8
            )
        if cutlass.const_expr(self.token_back_push_sf):
            token_back_sf = smem_workspace.tensor(self.token_back_sf_smem_region, smem_base)
            worker_sf_stage = token_back_sf[transfer_warp_idx, (None, None)]
            sf_chunk_bytes = (
                cute.cosize(worker_sf_stage) * int(self.combine_format.scale_dtype.width) // 8
            )
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(worker_mbarrier, 1)
        cute.arch.sync_warp()

        owned_sizes = smem_workspace.tensor(self.expert_sizes_smem_region, smem_base)
        if cutlass.const_expr(self.token_back_mode == "standalone_warps"):
            sizes_ready_barrier = pipeline.NamedBarrier(
                barrier_id=self.standalone_size_barrier_id,
                num_threads=2 * self.transfer_thread_count,
            )
            sizes_ready_barrier.arrive_and_wait()

        pool_expert_bases = self._router.pool_expert_base_tensor(self._device_workspace)
        token_metadata_pointer = self._router.token_src_metadata_pointer(self._device_workspace)
        fc2_done = self._device_workspace.ptr(self.fc2_done_region)
        if cutlass.const_expr(self.token_back_push_data):
            fc2_activation_pointer = self._device_workspace.ptr(self.fc2_activation_region)
            output_token_bytes = self.hidden_size * int(self.combine_format.act_dtype.width) // 8
            activation_chunk_count = ceil_div(output_token_bytes, activation_chunk_bytes)
            data_window_unit = activation_chunk_bytes * 2
            reuse_data_pacing_enabled = (
                self.token_back_mode == "reuse_dispatch_warps"
                and data_window_unit > self.minimum_pacing_window_cycles
            )
            # Preserve the empirical low:initial:high ratio of 1:2.5:5.
            data_average_window = Int32(data_window_unit)
            data_low_window = data_window_unit * 2 // 5
            data_high_window = data_window_unit * 2
        if cutlass.const_expr(self.token_back_push_sf):
            fc2_sf_pointer = self._device_workspace.ptr(self.fc2_activation_sf_region)
            output_sf_bytes = (
                self.combine_sf_hidden_padded * int(self.combine_format.scale_dtype.width) // 8
            )
            sf_chunk_count = ceil_div(output_sf_bytes, sf_chunk_bytes)
            sf_window_unit = ceil_div(sf_chunk_bytes * 2, 3)
            reuse_sf_pacing_enabled = (
                self.token_back_mode == "reuse_dispatch_warps"
                and sf_window_unit > self.minimum_pacing_window_cycles
            )
            sf_average_window = Int32(sf_window_unit)
            sf_low_window = sf_window_unit * 2 // 5
            sf_high_window = sf_window_unit * 2

        next_dense_token = global_worker_idx - global_worker_count
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):
            next_dense_token = Int32(0)
        next_dense_token = self.next_token(next_dense_token)
        expert_valid_begin = Int32(0)
        transfer_phase = Int32(0)

        iket.range_push("token_back.work")
        local_expert = Int32(0)
        while local_expert < Int32(self.experts_per_rank):
            expert_token_count = owned_sizes[local_expert]
            expert_valid_end = expert_valid_begin + expert_token_count
            if next_dense_token < expert_valid_end:
                token_tile_count = (
                    expert_token_count + Int32(self.tokens_per_fc1_ready_slot - 1)
                ) // Int32(self.tokens_per_fc1_ready_slot)
                completion_target = token_tile_count * Int32(self.fc2_done_signals_per_token_tile)
                iket.range_push("token_back.wait_fc2")
                while (
                    cute.arch.load(fc2_done + local_expert, Int32, sem="acquire", scope="gpu")
                    < completion_target
                ):
                    nanosleep(500)
                iket.range_pop()

            while next_dense_token < expert_valid_end:
                token_in_expert = next_dense_token - expert_valid_begin
                pool_token_idx = pool_expert_bases[local_expert] + token_in_expert
                metadata = TokenSrcMetadata.load(
                    token_metadata_pointer.toint()
                    + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                )
                destination_topk = metadata.src_topk
                if cutlass.const_expr(self.reduce_topk_in_kernel):
                    destination_topk = Int32(0)
                peer_offset = self._token_comm_args.peer_rank_ptr_mapper.map(
                    Int64(0), metadata.src_rank, Int64(0)
                )
                is_remote_token = metadata.src_rank != self._local_rank

                if cutlass.const_expr(self.token_back_push_data):
                    iket.range_push("token_back.push_data")
                    local_activation_address = fc2_activation_pointer.toint() + Int64(
                        pool_token_idx
                    ) * Int64(output_token_bytes)
                    remote_activation = cute.make_tensor(
                        cute.make_ptr(
                            self._token_comm_args.pre_reduced_activation.dtype,
                            self._token_comm_args.pre_reduced_activation.iterator.toint()
                            + peer_offset,
                            AddressSpace.gmem,
                            assumed_align=(
                                self._token_comm_args.pre_reduced_activation.iterator.max_alignment
                            ),
                        ),
                        self._token_comm_args.pre_reduced_activation.layout,
                    )
                    destination_row = remote_activation[
                        Int64(metadata.src_token), destination_topk, None
                    ]
                    for chunk_idx in cutlass.range_constexpr(activation_chunk_count):
                        chunk_byte_offset = Int64(chunk_idx * activation_chunk_bytes)
                        chunk_bytes_this_round = min(
                            activation_chunk_bytes,
                            output_token_bytes - chunk_idx * activation_chunk_bytes,
                        )
                        current_chunk_bytes = Int32(chunk_bytes_this_round)
                        current_window_unit = ceil_div(chunk_bytes_this_round * 2, 3)
                        stateless_data_pacing_enabled = (
                            self.token_back_mode != "reuse_dispatch_warps"
                            and current_window_unit > self.minimum_pacing_window_cycles
                        )
                        round_start_clock = Int64(0)
                        cute.arch.sync_warp()
                        if cutlass.const_expr(
                            reuse_data_pacing_enabled or stateless_data_pacing_enabled
                        ):
                            if is_remote_token:
                                round_start_clock = read_clock64()
                            else:
                                round_start_clock = round_start_clock
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                worker_mbarrier, current_chunk_bytes
                            )
                            tma_load_1d(
                                worker_activation_stage.iterator,
                                local_activation_address + chunk_byte_offset,
                                worker_mbarrier,
                                current_chunk_bytes,
                            )
                        cute.arch.mbarrier_wait(worker_mbarrier, transfer_phase)
                        destination_chunk = cute.make_ptr(
                            cutlass.Uint8,
                            destination_row.iterator.toint() + chunk_byte_offset,
                            AddressSpace.gmem,
                            assumed_align=16,
                        )
                        with cute.arch.elect_one():
                            if cutlass.const_expr(self.reduce_topk_in_kernel):
                                cp_reduce_async_bulk_add_bf16_s2g(
                                    destination_chunk,
                                    worker_activation_stage.iterator,
                                    current_chunk_bytes,
                                )
                            else:
                                cp_async_bulk_s2g(
                                    destination_chunk,
                                    worker_activation_stage.iterator,
                                    current_chunk_bytes,
                                )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                        transfer_phase = transfer_phase ^ Int32(1)
                        cute.arch.sync_warp()
                        if cutlass.const_expr(reuse_data_pacing_enabled):
                            if is_remote_token:
                                current_window = Int32(read_clock64() - round_start_clock)
                                data_average_window = self._adaptive_pace(
                                    data_average_window,
                                    current_window,
                                    data_low_window,
                                    data_high_window,
                                )
                        elif cutlass.const_expr(stateless_data_pacing_enabled):
                            if is_remote_token:
                                current_window = Int32(read_clock64() - round_start_clock)
                                self._stateless_pace(Int32(current_window_unit), current_window)
                    iket.range_pop()

                if cutlass.const_expr(self.token_back_push_sf):
                    iket.range_push("token_back.push_sf")
                    local_sf_address = fc2_sf_pointer.toint() + Int64(pool_token_idx) * Int64(
                        output_sf_bytes
                    )
                    remote_sf = cute.make_tensor(
                        cute.make_ptr(
                            self._token_comm_args.pre_reduced_activation_sf.dtype,
                            self._token_comm_args.pre_reduced_activation_sf.iterator.toint()
                            + peer_offset,
                            AddressSpace.gmem,
                            assumed_align=(
                                self._token_comm_args.pre_reduced_activation_sf.iterator.max_alignment
                            ),
                        ),
                        self._token_comm_args.pre_reduced_activation_sf.layout,
                    )
                    destination_sf_row = remote_sf[
                        Int64(metadata.src_token), destination_topk, None
                    ]
                    for chunk_idx in cutlass.range_constexpr(sf_chunk_count):
                        chunk_byte_offset = Int64(chunk_idx * sf_chunk_bytes)
                        chunk_bytes_this_round = min(
                            sf_chunk_bytes, output_sf_bytes - chunk_idx * sf_chunk_bytes
                        )
                        current_chunk_bytes = Int32(chunk_bytes_this_round)
                        current_window_unit = ceil_div(chunk_bytes_this_round * 2, 3)
                        stateless_sf_pacing_enabled = (
                            self.token_back_mode != "reuse_dispatch_warps"
                            and current_window_unit > self.minimum_pacing_window_cycles
                        )
                        round_start_clock = Int64(0)
                        cute.arch.sync_warp()
                        if cutlass.const_expr(
                            reuse_sf_pacing_enabled or stateless_sf_pacing_enabled
                        ):
                            if is_remote_token:
                                round_start_clock = read_clock64()
                            else:
                                round_start_clock = round_start_clock
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                worker_mbarrier, current_chunk_bytes
                            )
                            tma_load_1d(
                                worker_sf_stage.iterator,
                                local_sf_address + chunk_byte_offset,
                                worker_mbarrier,
                                current_chunk_bytes,
                            )
                        cute.arch.mbarrier_wait(worker_mbarrier, transfer_phase)
                        destination_chunk = cute.make_ptr(
                            cutlass.Uint8,
                            destination_sf_row.iterator.toint() + chunk_byte_offset,
                            AddressSpace.gmem,
                            assumed_align=16,
                        )
                        with cute.arch.elect_one():
                            cp_async_bulk_s2g(
                                destination_chunk, worker_sf_stage.iterator, current_chunk_bytes
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                        transfer_phase = transfer_phase ^ Int32(1)
                        cute.arch.sync_warp()
                        if cutlass.const_expr(reuse_sf_pacing_enabled):
                            if is_remote_token:
                                current_window = Int32(read_clock64() - round_start_clock)
                                sf_average_window = self._adaptive_pace(
                                    sf_average_window, current_window, sf_low_window, sf_high_window
                                )
                        elif cutlass.const_expr(stateless_sf_pacing_enabled):
                            if is_remote_token:
                                current_window = Int32(read_clock64() - round_start_clock)
                                self._stateless_pace(Int32(current_window_unit), current_window)
                    iket.range_pop()

                cute.arch.sync_warp()
                next_dense_token = self.next_token(next_dense_token)

            expert_valid_begin = expert_valid_end
            local_expert = local_expert + Int32(1)
        iket.range_pop()
        # with cute.arch.elect_one():
        #     cute.printf(" final data_average_window: {} ", data_average_window)

    @cute.jit
    def next_token(self, current_token: Int32) -> Int32:
        global_worker_count = self.promised_launchable_sm_count * self.transfer_warp_count
        schedule_counter = None
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):
            schedule_counter = self._device_workspace.ptr(self.token_back_schedule_region)
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):
            claimed_token = Int32(0)
            if self._lane_idx == Int32(0):
                claimed_token = cute.arch.atomic_add(
                    schedule_counter, Int32(1), sem="relaxed", scope="gpu"
                )
            return Int32(cute.arch.shuffle_sync(claimed_token, Int32(0)))
        return current_token + global_worker_count

    @cute.jit
    def reset_tail(self) -> None:
        """Reset communication state with the four token-in transfer warps."""
        transfer_warp_idx = self._transfer_warp_idx
        lane_idx = self._lane_idx
        transfer_thread_idx = transfer_warp_idx * Int32(32) + lane_idx
        iket.range_push("tail.nvlink_drain")
        self._nvlink_barrier.arrive_and_wait(
            self.transfer_thread_count,
            Int32(self.promised_launchable_sm_count),
            self._linear_cta_idx,
            transfer_thread_idx,
            prologue_grid_sync=True,
            epilogue_grid_sync=False,
        )
        iket.range_pop()
        total_reset_threads = self.promised_launchable_sm_count * self.transfer_thread_count
        global_reset_thread = (
            self._linear_cta_idx * Int32(self.transfer_thread_count) + transfer_thread_idx
        )
        iket.range_push("tail.reset_workspace")
        self._device_workspace.reset_tail_space("shared", global_reset_thread, total_reset_threads)
        self._device_workspace.reset_tail_space("local", global_reset_thread, total_reset_threads)
        iket.range_pop()
        iket.range_push("tail.nvlink_publish")
        self._nvlink_barrier.arrive_and_wait(
            self.transfer_thread_count,
            Int32(self.promised_launchable_sm_count),
            self._linear_cta_idx,
            transfer_thread_idx,
            prologue_grid_sync=True,
            epilogue_grid_sync=False,
        )
        iket.range_pop()
        if cutlass.const_expr(os.environ.get("MEGA_USE_NCU", "0") == "1"):
            iket.range_push("tail.ncu_finalize")
            self._nvlink_barrier.finalize(
                2,
                self.transfer_thread_count,
                Int32(self.promised_launchable_sm_count),
                self._linear_cta_idx,
                transfer_thread_idx,
            )
            iket.range_pop()


__all__ = ["TokenBackMode", "TokenBackScheduleMode", "TokenCommArgs", "TokenCommNonDeterministic"]
