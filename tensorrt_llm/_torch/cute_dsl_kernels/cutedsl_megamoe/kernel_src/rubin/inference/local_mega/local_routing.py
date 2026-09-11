# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing support for the Rubin Local MegaMoE kernel."""

import dataclasses
from typing import Callable, ClassVar, Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

from .....api import ImplDesc, KernelComponent, OptionalRequirement, ProblemDesc
from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.dsl_helpers import mark_alignment
from .....helpers.flag_batch import make_flag_batch_tracker
from .....helpers.iket_compat import iket
from .....helpers.ptx_helpers import cp_async_bulk_s2g, nanosleep, stg_b64, stg_f32, tma_load_1d
from .....helpers.smem_workspace import SmemWorkspace
from .....helpers.software_sync import SoftwareGridSync
from .....helpers.utils import ceil_div, round_up
from .....quant_def import QuantKind


@dataclasses.dataclass(frozen=True)
class RoutingMetadata:
    """One i64 routing record containing an i32 token and i32 top-k slot."""

    token: Int32
    topk_slot: Int32

    nbytes: ClassVar[int] = 8

    def pack(self) -> Int64:
        return (Int64(self.topk_slot) << Int64(32)) | (Int64(self.token) & Int64(0xFFFFFFFF))

    @staticmethod
    def _pointer(address: Union[cute.Pointer, Int64]) -> cute.Pointer:
        raw_address = address if isinstance(address, Int64) else address.toint()
        return cute.make_ptr(Int64, raw_address, AddressSpace.gmem, assumed_align=8)

    @classmethod
    def load(cls, address: Union[cute.Pointer, Int64]) -> "RoutingMetadata":
        packed = Int64(cute.arch.load(cls._pointer(address), Int64, scope="gpu"))
        return cls(
            token=Int32(packed & Int64(0xFFFFFFFF)),
            topk_slot=Int32(packed >> Int64(32)),
        )


@dataclasses.dataclass(frozen=True)
class LocalRoutingArgs:
    """Device views materialized inside the fused kernel region."""

    activation_sf: cute.Tensor

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "LocalRoutingArgs":
        if values:
            raise ValueError(f"LocalRoutingArgs expected no MLIR values, got {len(values)}.")
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


@cute.jit
def _smem_exclusive_prefix(
    input_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    warp_totals: cute.Tensor,
    block_thread_count: int,
    thread_idx: Int32,
    lane_idx: Int32,
    warp_idx: Int32,
) -> Int32:
    """Compute a CTA-wide exclusive prefix over an Int32 SMEM tensor."""
    num_elements = cute.size(input_tensor)
    scan_rows = num_elements // 4
    num_warps = block_thread_count // 32
    input_vectors = cute.make_tensor(
        input_tensor.iterator, cute.make_layout((scan_rows, 4), stride=(4, 1))
    )
    load_atom = _copy_atom(cutlass.Int32, 128)

    values = cute.make_rmem_tensor((4,), cutlass.Int32)
    carry = Int32(0)
    for segment in cutlass.range_constexpr(ceil_div(scan_rows, block_thread_count)):
        row = Int32(segment * block_thread_count) + thread_idx
        if row < Int32(scan_rows):
            row_slice = input_vectors[row, None]
            cute.copy(load_atom, mark_alignment(row_slice, 16), values)
        else:
            for element in cutlass.range_constexpr(4):
                values[element] = Int32(0)

        local_prefix = (
            Int32(0),
            values[0],
            values[0] + values[1],
            values[0] + values[1] + values[2],
        )
        lane_total = local_prefix[3] + values[3]
        inclusive = lane_total
        for step_log in cutlass.range_constexpr(5):
            step = Int32(1 << step_log)
            previous = Int32(cute.arch.shuffle_sync(inclusive, lane_idx - step))
            if lane_idx >= step:
                inclusive = inclusive + previous
        lane_base = inclusive - lane_total
        warp_total = Int32(cute.arch.shuffle_sync(inclusive, Int32(31)))
        if lane_idx == Int32(0):
            warp_totals[warp_idx] = warp_total
        cute.arch.sync_threads()

        region_total = Int32(0)
        if lane_idx < Int32(num_warps):
            region_total = warp_totals[lane_idx]
        inclusive_region = region_total
        for step_log in cutlass.range_constexpr(5):
            step = Int32(1 << step_log)
            previous = Int32(cute.arch.shuffle_sync(inclusive_region, lane_idx - step))
            if lane_idx >= step:
                inclusive_region = inclusive_region + previous
        warp_base = Int32(cute.arch.shuffle_sync(inclusive_region - region_total, warp_idx))
        segment_total = Int32(cute.arch.shuffle_sync(inclusive_region, Int32(31)))
        base = carry + warp_base + lane_base
        if row < Int32(scan_rows):
            first_element = row * Int32(4)
            for element in cutlass.range_constexpr(4):
                output_tensor[first_element + Int32(element)] = base + local_prefix[element]
        carry = carry + segment_total
        cute.arch.sync_threads()
    return carry


class _MetadataPushRouter(KernelComponent):
    """Sort routes into one local expert-grouped pool."""

    router_smem_limit_bytes: ClassVar[int] = 227 * 1024
    router_warps_per_cta: ClassVar[int] = 16

    sizes_region = "local_mega.routing.sizes"
    sizes_ready_region = "local_mega.routing.sizes_ready"
    metadata_ready_region = "local_mega.routing.metadata_ready"
    sorted_metadata_region = "local_mega.routing.sorted_metadata"
    sorted_scores_region = "local_mega.routing.sorted_scores"
    pool_expert_base_region = "local_mega.routing.pool_expert_base"
    token_src_metadata_region = "local_mega.routing.token_src_metadata"
    fc1_topk_scores_region = "local_mega.routing.fc1_topk_scores"
    source_expert_base_region = "local_mega.routing.source_expert_base"
    sorted_metadata_ready_region = "local_mega.routing.sorted_metadata_ready"
    push_table_ready_region = "local_mega.routing.push_table_ready"
    router_size_counter_region = "local_mega.routing.router_size_counter"
    router_histogram_done_region = "local_mega.routing.router_histogram_done"
    source_base_ready_region = "local_mega.routing.source_base_ready"

    router_data_histogram_region = "local_mega.routing.router_smem.data_histogram"
    router_data_prefix_region = "local_mega.routing.router_smem.data_prefix"
    router_data_warp_totals_region = "local_mega.routing.router_smem.data_warp_totals"
    router_data_sorted_region = "local_mega.routing.router_smem.data_sorted"
    router_data_base_region = "local_mega.routing.router_smem.data_base"
    router_helper_totals_region = "local_mega.routing.router_smem.helper_totals"
    router_helper_prefix_region = "local_mega.routing.router_smem.helper_prefix"
    router_helper_warp_totals_region = "local_mega.routing.router_smem.helper_warp_totals"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "expert_count": int,
            "topk": int,
            "max_tokens": int,
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

        self.expert_count = problem_desc["expert_count"]
        self.topk = problem_desc["topk"]
        self.max_tokens = problem_desc["max_tokens"]
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
        self._router_smem_workspace = self._build_router_smem_workspace()

        self._device_workspace = None
        self._router_thread_idx = None
        self._router_linear_cta_idx = None
        self._router_grid_thread_idx = None
        self._router_warp_idx = None
        self._router_lane_idx = None

    def _validate_router_configuration(self) -> None:
        positive_fields = (
            "expert_count",
            "topk",
            "max_tokens",
            "token_padding_block",
            "promised_launchable_sm_count",
            "router_smem_limit_bytes",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")
        if self.expert_count > 16384:
            raise NotImplementedError("Local routing supports at most 16384 experts.")
        if self.topk > self.expert_count:
            raise ValueError(
                f"topk must not exceed expert_count, got {self.topk} and {self.expert_count}."
            )

    def worst_case_padded_tokens(self, block: int) -> int:
        source_token_capacity = self.max_tokens
        routes_per_source_token = min(self.topk, self.expert_count)
        route_capacity = source_token_capacity * routes_per_source_token
        active_expert_capacity = min(self.expert_count, route_capacity)
        route_budget_blocks = (
            active_expert_capacity + (route_capacity - active_expert_capacity) // block
        )
        expert_bound_blocks = active_expert_capacity * int(ceil_div(source_token_capacity, block))
        return min(route_budget_blocks, expert_bound_blocks) * block

    @property
    def worst_case_token_count(self) -> int:
        return self.worst_case_padded_tokens(self.token_padding_block)

    def _router_launch_configuration(self) -> Tuple[int, int]:
        def next_power_of_two(value: int) -> int:
            return 1 << (max(value, 1) - 1).bit_length()

        routed_token_capacity = self.max_tokens * self.topk
        minimum_cta_capacity = 2048
        maximum_cta_capacity = 16384
        maximum_data_cta_count = 128
        maximum_supported_tokens = maximum_cta_capacity * maximum_data_cta_count
        if routed_token_capacity > maximum_supported_tokens:
            raise NotImplementedError(
                f"The router supports at most {maximum_supported_tokens} routed tokens."
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
        overlay = workspace.create_overlay("local_mega.routing.router_smem.role")
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
        maximum_routed_tokens = self.max_tokens * self.topk
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
            self.pool_expert_base_region, cutlass.Int32, (self.expert_count,), buffer_space="local"
        )
        workspace.register(
            self.source_expert_base_region,
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
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        device_workspace: DeviceWorkspace,
        stream: cuda.CUstream,
    ) -> None:
        """Launch counting-sort DATA, prefix HELPER, and metadata placement roles."""
        if cutlass.const_expr(self.apply_topk_at_fc1 and topk_scores is None):
            raise ValueError("apply_topk_at_fc1 requires router topk_scores.")
        self._router_kernel(
            topk_indices,
            topk_scores,
            local_workspace,
            shared_workspace,
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
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
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
            publish_sizes = self._publish_sizes(
                cute.make_tensor(histogram.iterator, cute.make_layout((self.expert_count_padded,)))
            )
            total_valid_routes = _smem_exclusive_prefix(
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

            total_valid_routes = _smem_exclusive_prefix(
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
        iket.range_push("router.compute_local_tables")
        self._compute_local_tables(smem_base)
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

        iket.range_push("router.publish_sizes")
        for expert_round in cutlass.range_constexpr(expert_round_count):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if expert < Int32(self.expert_count_padded):
                totals[expert] = size_counter[expert]
        cute.arch.sync_threads()

        publish_sizes = self._publish_sizes(totals)
        _smem_exclusive_prefix(
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

        iket.range_push("router.compute_local_tables")
        self._compute_local_tables(smem_base)
        iket.range_pop()

    @cute.jit
    def _router_push_metadata(self) -> None:
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
            sizes = self._device_workspace.tensor(self.sizes_region)
            source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
            pool_expert_base = self._device_workspace.tensor(self.pool_expert_base_region)
            route_count = sizes[global_expert]
            source_begin = source_expert_base[global_expert]
            destination_begin = pool_expert_base[global_expert]
            source_metadata = self._device_workspace.ptr(self.sorted_metadata_region)
            destination_metadata_address = self._device_workspace.ptr(
                self.token_src_metadata_region
            ).toint()
            if cutlass.const_expr(self.apply_topk_at_fc1):
                source_scores = self._device_workspace.ptr(self.sorted_scores_region)
                destination_scores_address = self._device_workspace.ptr(
                    self.fc1_topk_scores_region
                ).toint()
            route_round_count = (route_count + Int32(31)) // Int32(32)
            for route_round in cutlass.range(route_round_count, unroll=1):
                route = Int32(route_round) * Int32(32) + self._router_lane_idx
                if route < route_count:
                    source_position = source_begin + route
                    destination_position = destination_begin + route
                    metadata = cute.arch.load(source_metadata + source_position, cutlass.Int64)
                    stg_b64(
                        destination_metadata_address
                        + Int64(destination_position) * Int64(RoutingMetadata.nbytes),
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
            cute.arch.fence_acq_rel_gpu()
        # Keep the notifier behind the leader's device fence.
        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            cute.arch.atomic_add(
                self._device_workspace.ptr(self.metadata_ready_region),
                Int32(1),
                sem="release",
                scope="gpu",
            )

    @cute.jit
    def _publish_sizes(self, smem_expert_counts: cute.Tensor) -> Callable[[], None]:
        row_bytes = Int32(self.expert_count_padded * 4)
        if self._router_warp_idx == Int32(0):
            with cute.arch.elect_one():
                cp_async_bulk_s2g(
                    self._device_workspace.ptr(self.sizes_region),
                    smem_expert_counts.iterator,
                    row_bytes,
                )
                cute.arch.cp_async_bulk_commit_group()

        def finalize() -> None:
            cute.arch.cp_async_bulk_wait_group(0)
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.fence_acq_rel_gpu()
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.sizes_ready_region),
                    Int32(1),
                    sem="release",
                    scope="gpu",
                )

        return finalize

    @cute.jit
    def _compute_local_tables(self, smem_base: cute.Pointer) -> None:
        block_thread_count = self.router_warps_per_cta * 32

        padded_totals = self._router_smem_workspace.tensor(
            self.router_helper_totals_region, smem_base
        )
        prefix = self._router_smem_workspace.tensor(self.router_helper_prefix_region, smem_base)
        warp_totals = self._router_smem_workspace.tensor(
            self.router_helper_warp_totals_region, smem_base
        )
        sizes = self._device_workspace.tensor(self.sizes_region)

        sizes_ready = self._device_workspace.ptr(self.sizes_ready_region)
        iket.range_push("router.wait_sizes_ready")
        if self._router_thread_idx == Int32(0):
            while cute.arch.load(sizes_ready, Int32, sem="acquire", scope="gpu") != Int32(1):
                nanosleep(150)
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("router.compute_padded_prefix")
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
        _smem_exclusive_prefix(
            padded_totals,
            prefix,
            warp_totals,
            block_thread_count,
            self._router_thread_idx,
            self._router_lane_idx,
            self._router_warp_idx,
        )
        pool_expert_base = self._device_workspace.tensor(self.pool_expert_base_region)
        for expert_round in cutlass.range_constexpr(padded_expert_rounds):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if expert < Int32(self.expert_count):
                pool_expert_base[expert] = prefix[expert]
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
        maximum_elements = self.max_tokens * self.topk
        actual_token_count = Int32(self.max_tokens)
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
            metadata = RoutingMetadata(
                token=(element.flat_topk_index // Int32(self.topk)),
                topk_slot=(element.flat_topk_index % Int32(self.topk)),
            )
            stg_b64(
                metadata_address + Int64(position) * Int64(RoutingMetadata.nbytes),
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
                    metadata = RoutingMetadata(
                        token=(element.flat_topk_index // Int32(self.topk)),
                        topk_slot=(element.flat_topk_index % Int32(self.topk)),
                    )
                    stg_b64(
                        metadata_address + Int64(output_position) * Int64(RoutingMetadata.nbytes),
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
    def expert_sizes(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        """Return the local expert-size vector."""
        sizes = device_workspace.tensor(self.sizes_region)
        return cute.make_tensor(sizes.iterator, cute.make_layout((self.expert_count,)))

    @property
    def metadata_ready_target(self) -> int:
        return self.router_push_cta_count

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
            while cute.arch.load(sizes_ready, Int32, sem="acquire", scope="gpu") != Int32(1):
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
            while cute.arch.load(metadata_ready, Int32, sem="acquire", scope="gpu") != Int32(
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


class LocalRouting(KernelComponent):
    """Build expert-grouped routing state for the fused local kernel."""

    transfer_warp_count: ClassVar[int] = 4
    transfer_thread_count: ClassVar[int] = transfer_warp_count * 32
    grid_sync_barrier_id: ClassVar[int] = 10
    token_in_size_barrier_id: ClassVar[int] = 12

    fc1_ready_region = "local_mega.routing.fc1_ready"
    fc1_activation_sf_region = "local_mega.routing.fc1_activation_sf"
    pre_reduced_activation_region = "local_mega.routing.pre_reduced_activation"

    token_in_mbarrier_region = "local_mega.routing.main_smem.token_in_mbarriers"
    expert_sizes_smem_region = "local_mega.routing.main_smem.expert_sizes"
    token_in_sf_smem_region = "local_mega.routing.main_smem.token_in_sf"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "expert_count": int,
            "topk": int,
            "max_tokens": int,
            "hidden_size": int,
            "apply_topk_at_fc1": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "token_padding_block": int,
            "sf_padding_block": int,
            "tokens_per_fc1_ready_slot": int,
            "promised_launchable_sm_count": int,
            "token_in_flag_batch": int,
            "reduce_topk_in_kernel": bool,
            "router_smem_limit_bytes": OptionalRequirement(int),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.expert_count = problem_desc["expert_count"]
        self.topk = problem_desc["topk"]
        self.max_tokens = problem_desc["max_tokens"]
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind.nvfp4
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.tokens_per_fc1_ready_slot = impl_desc["tokens_per_fc1_ready_slot"]
        self.promised_launchable_sm_count = impl_desc["promised_launchable_sm_count"]
        self.token_in_flag_batch = impl_desc["token_in_flag_batch"]
        self.reduce_topk_in_kernel = impl_desc["reduce_topk_in_kernel"]
        self.router_smem_limit_bytes = impl_desc.get("router_smem_limit_bytes", 227 * 1024)

        self._validate_configuration()
        self._router = _MetadataPushRouter(problem_desc, impl_desc)
        self._grid_sync = SoftwareGridSync(barrier_id=self.grid_sync_barrier_id)
        self._device_workspace = None
        self._local_routing_args = None
        self._linear_cta_idx = None
        self._transfer_warp_idx = None
        self._lane_idx = None

    def _validate_configuration(self) -> None:
        positive_fields = (
            "expert_count",
            "topk",
            "max_tokens",
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
        if self.expert_count > 16384:
            raise NotImplementedError("Local routing supports at most 16384 experts.")
        if self.topk > self.expert_count:
            raise ValueError(
                f"topk must not exceed expert_count, got {self.topk} and {self.expert_count}."
            )
        if not 1 <= self.token_in_flag_batch <= 32:
            raise ValueError(
                f"token_in_flag_batch must be in [1, 32], got {self.token_in_flag_batch}."
            )
        if self.tokens_per_fc1_ready_slot % self.token_padding_block != 0:
            raise ValueError("tokens_per_fc1_ready_slot must be divisible by token_padding_block.")
        element_block = self.activation_sf_vector_size * 4
        if self.hidden_size % element_block != 0:
            raise ValueError(
                f"{self.quant_kind} requires hidden_size divisible by {element_block}."
            )
        if self.sf_padding_block % 128 != 0:
            raise ValueError("sf_padding_block must be a multiple of 128.")

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
    def activation_sf_hidden_padded(self) -> int:
        valid_hidden = self.hidden_size // self.activation_sf_vector_size
        elements_per_16_bytes = 128 // int(self.activation_sf_dtype.width)
        return int(round_up(valid_hidden, elements_per_16_bytes))

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
    def token_src_metadata_region(self) -> str:
        return self._router.token_src_metadata_region

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        self._router.register_device_workspace(workspace)
        self._register_main_workspace(workspace)
        self._grid_sync.register_device_workspace(workspace)

    @cute.jit
    def launch_router(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        device_workspace: DeviceWorkspace,
        stream: cuda.CUstream,
    ) -> None:
        self._router.launch_router(
            topk_indices,
            topk_scores,
            local_workspace,
            shared_workspace,
            device_workspace,
            stream,
        )

    @cute.jit
    def expert_sizes(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return self._router.expert_sizes(device_workspace)

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
        local_routing_args: LocalRoutingArgs,
        linear_cta_idx: Int32,
    ) -> None:
        self._device_workspace = device_workspace
        self._local_routing_args = local_routing_args
        self._linear_cta_idx = linear_cta_idx
        thread_idx, _, _ = cute.arch.thread_idx()
        transfer_thread_idx = thread_idx % Int32(self.transfer_thread_count)
        self._transfer_warp_idx = cute.arch.make_warp_uniform(transfer_thread_idx // Int32(32))
        self._lane_idx = transfer_thread_idx % Int32(32)
        self._grid_sync.assign_device_members(device_workspace)

    def remove_device_members(self) -> None:
        self._grid_sync.remove_device_members()
        self._device_workspace = None
        self._local_routing_args = None
        self._linear_cta_idx = None
        self._transfer_warp_idx = None
        self._lane_idx = None

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "LocalRouting":
        if values:
            raise ValueError("LocalRouting carries no MLIR values.")
        return self

    def _register_main_workspace(self, workspace: DeviceWorkspace) -> None:
        """Register the fused kernel's FC1 scale pool and optional combine staging.

        FC1 scale pool: ``token_in`` only materializes activation scale factors in
        expert-grouped POOL index space.  The activation payload is intentionally
        not duplicated; the Local MegaMoE FC1 mainloop gathers it directly from the caller
        tensor into B SMEM using ``token_src_metadata``.

        Combine staging, iff ``not reduce_topk_in_kernel``: the BF16 per-topk
        output plane addressed as ``(token, topk_slot, hidden)``. Every exposed
        cell is rewritten each launch, so the plane does not require reset.
        """
        workspace.register(
            self.fc1_ready_region,
            cutlass.Int32,
            (self.max_fc1_ready_slot_count,),
            buffer_space="local",
            reset="tail_reset",
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

        if not self.reduce_topk_in_kernel:
            workspace.register(
                self.pre_reduced_activation_region,
                cutlass.BFloat16,
                (self.max_tokens, self.topk, self.hidden_size),
                buffer_space="shared",
                mem_order=(2, 1, 0),
                byte_alignment=128,
            )

    def register_smem_regions(self, workspace: SmemWorkspace) -> None:
        workspace.register_mbarrier(self.token_in_mbarrier_region, self.transfer_warp_count)
        workspace.register_tensor(
            self.expert_sizes_smem_region, cutlass.Int32, (self.expert_count,), byte_alignment=16
        )
        transfer_overlay = workspace.create_overlay("local_mega.routing.main_smem.transfer")
        token_in_lifetime = transfer_overlay.add_lifetime("token_in")
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

    @cute.jit
    def fc1_ready_counter_pointer(self, device_workspace: DeviceWorkspace) -> cute.Pointer:
        return device_workspace.ptr(self.fc1_ready_region)

    @cute.jit
    def fc1_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        layout = tile_atom_to_shape_SF(
            (self.worst_case_sf_token_count, self.hidden_size, 1), self.activation_sf_vector_size
        )
        return cute.make_tensor(
            device_workspace.ptr(self.fc1_activation_sf_region), cute.select(layout, mode=[0, 1])
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
    def token_in(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """Wait for pushed metadata, then materialize only activation scales.

        Local MegaMoE gathers the activation payload directly into FC1 B SMEM, so this
        path deliberately avoids the former GMEM -> SMEM -> expert-pool copy.
        """
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
        source_sizes = cute.make_tensor(sizes.iterator, cute.make_layout((self.expert_count,)))
        copy_elements = 4 if self.expert_count % 4 == 0 else 1
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
        size_copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL
                if copy_elements == 4
                else cute.nvgpu.LoadCacheMode.ALWAYS
            ),
            cutlass.Int32,
            num_bits_per_copy=copy_elements * 32,
        )
        size_copy_rounds = ceil_div(size_vector_count, self.transfer_thread_count)
        transfer_thread_idx = transfer_warp_idx * Int32(32) + lane_idx
        for size_copy_round in cutlass.range_constexpr(size_copy_rounds):
            vector_idx = Int32(size_copy_round * self.transfer_thread_count) + transfer_thread_idx
            if vector_idx < Int32(size_vector_count):
                cute.copy(
                    size_copy_atom,
                    source_size_vectors[None, vector_idx],
                    destination_size_vectors[None, vector_idx],
                )
        cute.arch.cp_async_commit_group()
        iket.range_pop()

        iket.range_push("token_in.wait_metadata_ready")
        self._router.wait_for_metadata_ready(self._device_workspace)
        iket.range_pop()

        cute.arch.cp_async_wait_group(0)
        iket.range_push("token_in.size_barrier")
        token_in_size_barrier = pipeline.NamedBarrier(
            barrier_id=self.token_in_size_barrier_id, num_threads=self.transfer_thread_count
        )
        token_in_size_barrier.arrive_and_wait()
        iket.range_pop()

        iket.range_push("token_in.pull_payload")
        token_in_mbarriers = smem_workspace.ptr(self.token_in_mbarrier_region, smem_base)
        token_in_sf = smem_workspace.tensor(self.token_in_sf_smem_region, smem_base)
        warp_mbarrier = token_in_mbarriers + transfer_warp_idx
        warp_sf_stage = token_in_sf[transfer_warp_idx, (None, None)]
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(warp_mbarrier, 1)
        cute.arch.sync_warp()

        fc1_activation_sf = self.fc1_activation_sf_tensor(self._device_workspace)
        fc1_ready_counter = self._device_workspace.ptr(self.fc1_ready_region)
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
        while local_expert < Int32(self.expert_count):
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

                metadata = RoutingMetadata.load(
                    token_metadata_pointer.toint()
                    + Int64(pool_token_idx) * Int64(RoutingMetadata.nbytes)
                )
                source_sf_row = self._local_routing_args.activation_sf[Int64(metadata.token), None]

                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        warp_mbarrier, Int32(activation_sf_bytes)
                    )
                    tma_load_1d(
                        warp_sf_stage.iterator,
                        source_sf_row.iterator,
                        warp_mbarrier,
                        Int32(activation_sf_bytes),
                    )
                cute.arch.sync_warp()
                cute.arch.mbarrier_wait(warp_mbarrier, pull_phase)

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

    @cute.jit
    def reset_tail(self) -> None:
        """Reset persistent-kernel tail state with the four token-in transfer warps."""
        transfer_warp_idx = self._transfer_warp_idx
        lane_idx = self._lane_idx
        transfer_thread_idx = transfer_warp_idx * Int32(32) + lane_idx
        iket.range_push("tail.grid_sync_before_reset")
        self._grid_sync.sync(
            self.transfer_thread_count,
            Int32(self.promised_launchable_sm_count),
            self._linear_cta_idx,
            transfer_thread_idx,
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
        iket.range_push("tail.grid_sync_after_reset")
        self._grid_sync.sync(
            self.transfer_thread_count,
            Int32(self.promised_launchable_sm_count),
            self._linear_cta_idx,
            transfer_thread_idx,
        )
        iket.range_pop()


__all__ = ["LocalRouting", "LocalRoutingArgs"]
