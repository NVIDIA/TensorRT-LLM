"""Metadata-push routing with a fixed token sequence and fused communication."""

import dataclasses
import os
from typing import Callable, ClassVar, Optional, Tuple, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

from ...api import ImplDesc, KernelComponent, OptionalRequirement, ProblemDesc
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.dsl_helpers import smem_exclusive_prefix
from ...helpers.flag_batch import make_flag_batch_tracker
from ...helpers.iket_compat import iket
from ...helpers.software_sync import NvlinkBarrier
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
from ...helpers.utils import ceil_div, round_up
from ...quant_def import CombineFormat, QuantKind
from ..token_protocol import TokenSrcMetadata
from .symmetric_buffer import SymmetricBufferDevice
from .token_comm import TokenBackMode, TokenBackScheduleMode, TokenCommArgs


_quant_spec = {
    "bf16": (cutlass.BFloat16, None, 0),
    "nvfp4": (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
    "mxfp4": (cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
    "mxfp8_e4m3": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
    "mxfp8_e5m2": (cutlass.Float8E5M2, cutlass.Float8E8M0FNU, 32),
}


@dataclasses.dataclass(frozen=True)
class _ReceiveCapacity:
    raw_route_count: int
    logical_route_count: int
    padded_route_count: int


def _compute_receive_capacity(
    *,
    world_size: int,
    max_tokens_per_rank: int,
    topk: int,
    experts_per_rank: int,
    max_recv_size_per_rank: int,
    padding_block: int,
) -> _ReceiveCapacity:
    raw_route_count = world_size * max_tokens_per_rank * topk
    logical_route_count = min(max_recv_size_per_rank, raw_route_count)
    active_expert_count = min(experts_per_rank, logical_route_count)
    padded_block_count = active_expert_count + (logical_route_count - active_expert_count) // padding_block
    return _ReceiveCapacity(
        raw_route_count=raw_route_count,
        logical_route_count=logical_route_count,
        padded_route_count=padded_block_count * padding_block,
    )


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
        return cls(flat_topk_index=scratch[0], topk_score=cute.recast_tensor(scratch, cutlass.Float32)[1])


@cute.jit
def _copy_atom(dtype, num_bits_per_copy: int):
    return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=num_bits_per_copy)


@cute.jit
def _mark_alignment(tensor: cute.Tensor, byte_alignment: int) -> cute.Tensor:
    pointer = tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(pointer.dtype, pointer.toint(), pointer.memspace, assumed_align=byte_alignment), tensor.layout
    )


@cute.jit
def _device_trap() -> None:
    """Terminate the current kernel after its failure state is published."""
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="trap;",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
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
    token_src_metadata_region = "nvlink.token_comm.token_src_metadata"  # nosec B105
    fc1_topk_scores_region = "nvlink.token_comm.fc1_topk_scores"
    source_expert_base_region = "nvlink.token_comm.source_expert_base"
    push_destination_base_region = "nvlink.token_comm.push_destination_base"
    sorted_metadata_ready_region = "nvlink.token_comm.sorted_metadata_ready"
    # Private continuation: zero means this (source token, topk slot) is absent.
    input_dedup_routes_region = "nvlink.token_comm.input_dedup_routes"
    push_table_ready_region = "nvlink.token_comm.push_table_ready"
    router_histogram_done_region = "nvlink.token_comm.router_histogram_done"
    router_cta_histograms_region = "nvlink.token_comm.router_cta_histograms"
    source_base_ready_region = "nvlink.token_comm.source_base_ready"

    router_data_histogram_region = "nvlink.token_comm.router_smem.data_histogram"
    router_data_totals_region = "nvlink.token_comm.router_smem.data_totals"
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
            "max_recv_size_per_rank": int,
            "apply_topk_at_fc1": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {"token_padding_block": int, "promised_launchable_sm_count": int, "drop_on_overflow": bool,
                "input_dedup_mode": OptionalRequirement(str),
                "interleave_source_ranks": OptionalRequirement(bool),
                "sf_padding_block": OptionalRequirement(int),
                "tokens_per_fc1_ready_slot": OptionalRequirement(int)}

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.world_size = problem_desc["world_size"]
        self.expert_count = problem_desc["expert_count"]
        self.topk = problem_desc["topk"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.max_recv_size_per_rank = min(
            problem_desc["max_recv_size_per_rank"], self.world_size * self.max_tokens_per_rank * self.topk
        )
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.token_padding_block = impl_desc["token_padding_block"]
        self.promised_launchable_sm_count = impl_desc["promised_launchable_sm_count"]
        self.drop_on_overflow = impl_desc["drop_on_overflow"]
        self.interleave_source_ranks = impl_desc.get("interleave_source_ranks", False)

        self.input_dedup_mode = impl_desc.get("input_dedup_mode", "off")
        if self.input_dedup_mode not in ("off", "rank"):
            raise ValueError("input_dedup_mode must be off/rank")
        # The retained specialization uses expert-major ownership and publishes
        # readiness only after every local fanout store has completed.
        self.input_dedup_store_mode = "batched" if self.input_dedup_mode != "off" else "serial"
        self.input_dedup_order = "expert" if self.input_dedup_mode != "off" else "token"
        self._validate_router_configuration()
        token_capacity = self.receive_capacity(self.token_padding_block)
        if self.input_dedup_mode != "off":
            # Route encoding uses three 20-bit row/ready fields. The smallest
            # destination row selects the owner without encoding an expert ID.
            if not self.apply_topk_at_fc1:
                raise ValueError("Input dedup requires backward routing")
            self.sf_padding_block = impl_desc["sf_padding_block"]
            self.tokens_per_fc1_ready_slot = impl_desc["tokens_per_fc1_ready_slot"]
            if self.sf_padding_block <= 0 or self.tokens_per_fc1_ready_slot <= 0:
                raise ValueError("Input dedup requires positive SF padding and ready-slot granularity")
            if self.max_recv_size_per_rank != self.world_size * self.max_tokens_per_rank * self.topk:
                raise ValueError("Input dedup requires uncapped receive capacity")
            # The low field stores pool_row + 1; zero denotes an unused route.
            if token_capacity.padded_route_count >= (1 << 20):
                raise ValueError("Input dedup requires padded receive capacity < 2**20 rows")
            # SF rows and ready slots have separate zero-based 20-bit fields.
            sf_capacity = self.receive_capacity(self.sf_padding_block).padded_route_count
            ready_slots = (
                self.receive_capacity(self.tokens_per_fc1_ready_slot).padded_route_count
                // self.tokens_per_fc1_ready_slot
            )
            if sf_capacity > (1 << 20):
                raise ValueError("Input dedup requires padded SF capacity <= 2**20 rows")
            if ready_slots > (1 << 20):
                raise ValueError("Input dedup requires at most 2**20 ready slots")
        self.raw_route_count = token_capacity.raw_route_count
        self.logical_route_capacity = token_capacity.logical_route_count
        self.worst_case_token_count = token_capacity.padded_route_count
        self.expert_count_padded = round_up(self.expert_count, 4)
        self.expert_count_with_trash = self.expert_count_padded + 1
        self.router_elements_per_lane, self.router_data_cta_count = self._router_launch_configuration()
        self.router_tokens_per_cta = self.router_elements_per_lane * self.router_warps_per_cta * 32
        self.router_push_cta_count = ceil_div(self.expert_count, self.router_warps_per_cta)
        self.router_grid_cta_count = max(self.router_data_cta_count + 1, self.router_push_cta_count)
        if self.router_grid_cta_count > self.promised_launchable_sm_count:
            raise ValueError(
                "Router grid exceeds promised_launchable_sm_count; all metadata-push CTAs must be concurrently resident."
            )
        self._router_smem_workspace = self._build_router_smem_workspace()

        self._device_workspace = None
        self._peer_rank_ptr_mapper = None
        self._router_local_rank = None
        self._router_thread_idx = None
        self._router_linear_cta_idx = None
        self._router_grid_thread_idx = None
        self._router_warp_idx = None
        self._router_lane_idx = None
        self._overflow_flag = None

    def _validate_router_configuration(self) -> None:
        if type(self.max_recv_size_per_rank) is not int:
            raise TypeError(f"max_recv_size_per_rank must be an int, got {type(self.max_recv_size_per_rank).__name__}.")
        if type(self.drop_on_overflow) is not bool:
            raise TypeError(f"drop_on_overflow must be a bool, got {type(self.drop_on_overflow).__name__}.")
        positive_fields = (
            "world_size",
            "expert_count",
            "topk",
            "max_tokens_per_rank",
            "max_recv_size_per_rank",
            "token_padding_block",
            "promised_launchable_sm_count",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")
        if self.max_tokens_per_rank > TokenSrcMetadata.MAX_TOKENS:
            raise ValueError(
                f"TokenSrcMetadata keeps the token index in {TokenSrcMetadata.TOKEN_BITS} bits; "
                f"max_tokens_per_rank must be <= {TokenSrcMetadata.MAX_TOKENS}, got {self.max_tokens_per_rank}."
            )
        if self.expert_count % self.world_size != 0:
            raise ValueError(
                f"expert_count must be divisible by world_size, got {self.expert_count} and {self.world_size}."
            )
        if self.expert_count > 16384:
            raise NotImplementedError("TokenComm supports at most 16384 global experts.")
        if self.topk > self.expert_count:
            raise ValueError(f"topk must not exceed expert_count, got {self.topk} and {self.expert_count}.")

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.world_size

    def receive_capacity(self, padding_block: int) -> _ReceiveCapacity:
        return _compute_receive_capacity(
            world_size=self.world_size,
            max_tokens_per_rank=self.max_tokens_per_rank,
            topk=self.topk,
            experts_per_rank=self.experts_per_rank,
            max_recv_size_per_rank=self.max_recv_size_per_rank,
            padding_block=padding_block,
        )

    def _router_launch_configuration(self) -> Tuple[int, int]:
        def next_power_of_two(value: int) -> int:
            return 1 << (max(value, 1) - 1).bit_length()

        routed_token_capacity = self.max_tokens_per_rank * self.topk
        minimum_cta_capacity = 2048
        maximum_cta_capacity = 16384
        maximum_data_cta_count = 128
        maximum_supported_tokens = maximum_cta_capacity * maximum_data_cta_count
        if routed_token_capacity > maximum_supported_tokens:
            raise NotImplementedError(f"The router supports at most {maximum_supported_tokens} routed tokens per rank.")
        cta_capacity = min(maximum_cta_capacity, next_power_of_two(max(routed_token_capacity, minimum_cta_capacity)))
        elements_per_lane = cta_capacity // (self.router_warps_per_cta * 32)
        data_cta_count = ceil_div(routed_token_capacity, cta_capacity)
        return elements_per_lane, data_cta_count

    def _build_router_smem_workspace(self) -> SmemWorkspace:
        workspace = SmemWorkspace()
        workspace.register_mbarrier(self.router_helper_load_mbarrier_region, 1)
        overlay = workspace.create_overlay("nvlink.token_comm.router_smem.role")
        data_lifetime = overlay.add_lifetime("data_cta")
        data_lifetime.register_tensor(
            self.router_data_histogram_region,
            cutlass.Int32,
            (self.expert_count_with_trash, self.router_warps_per_cta + 1),
            stride=(self.router_warps_per_cta + 1, 1),
        )
        data_lifetime.register_tensor(
            self.router_data_totals_region, cutlass.Int32, (self.expert_count_with_trash,), byte_alignment=16
        )
        data_lifetime.register_tensor(
            self.router_data_prefix_region, cutlass.Int32, (self.expert_count_with_trash,), byte_alignment=16
        )
        data_lifetime.register_tensor(self.router_data_warp_totals_region, cutlass.Int32, (self.router_warps_per_cta,))
        data_lifetime.register_tensor(
            self.router_data_sorted_region,
            (cutlass.Int64 if self.apply_topk_at_fc1 else cutlass.Int32),
            (self.router_tokens_per_cta,),
            byte_alignment=16,
        )
        if self.router_data_cta_count > 1:
            data_lifetime.register_tensor(
                self.router_data_base_region, cutlass.Int32, (self.expert_count_padded,), byte_alignment=16
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
            self.router_helper_totals_region, cutlass.Int32, (self.expert_count_padded,), byte_alignment=16
        )
        helper_lifetime.register_tensor(
            self.router_helper_prefix_region, cutlass.Int32, (self.expert_count_padded,), byte_alignment=16
        )
        helper_lifetime.register_tensor(
            self.router_helper_warp_totals_region, cutlass.Int32, (self.router_warps_per_cta,), byte_alignment=16
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
        if self.input_dedup_mode != "off":
            workspace.register(
                self.input_dedup_routes_region, cutlass.Int64,
                (self.world_size * maximum_routed_tokens,),
                buffer_space="shared", byte_alignment=128, reset="tail_reset",
            )
        workspace.register(
            self.sizes_by_rank_region,
            cutlass.Int32,
            (self.world_size, self.expert_count_padded),
            buffer_space="shared",
            stride=(self.expert_count_padded, 1),
        )
        workspace.register(
            self.sizes_region, cutlass.Int32, (self.expert_count_padded,), buffer_space="shared", reset="tail_reset"
        )
        workspace.register(self.sizes_ready_region, cutlass.Int32, (1,), buffer_space="shared", reset="tail_reset")
        workspace.register(self.metadata_ready_region, cutlass.Int32, (1,), buffer_space="shared", reset="tail_reset")
        workspace.register(self.sorted_metadata_region, cutlass.Int64, (maximum_routed_tokens,), buffer_space="local")
        if self.apply_topk_at_fc1:
            workspace.register(
                self.sorted_scores_region, cutlass.Float32, (maximum_routed_tokens,), buffer_space="local"
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
                self.fc1_topk_scores_region, cutlass.Float32, (self.worst_case_token_count,), buffer_space="shared"
            )
        workspace.register(self.pool_expert_base_region, cutlass.Int32, (self.experts_per_rank,), buffer_space="local")
        workspace.register(
            self.source_expert_base_region, cutlass.Int32, (self.expert_count_padded,), buffer_space="local"
        )
        workspace.register(
            self.push_destination_base_region, cutlass.Int32, (self.expert_count_padded,), buffer_space="local"
        )
        workspace.register(
            self.sorted_metadata_ready_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
        )
        workspace.register(self.push_table_ready_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset")
        if self.router_data_cta_count > 1:
            workspace.register(
                self.router_cta_histograms_region,
                cutlass.Int32,
                (self.router_data_cta_count, self.expert_count_padded),
                buffer_space="local",
                stride=(self.expert_count_padded, 1),
            )
            workspace.register(
                self.router_histogram_done_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
            )
            workspace.register(
                self.source_base_ready_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
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
        overflow_flag: cute.Tensor,
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
        if cutlass.const_expr(overflow_flag.iterator.dtype is not cutlass.Int32):
            raise TypeError("overflow_flag must use Int32 elements.")
        if cutlass.const_expr(cute.size(overflow_flag) != 1):
            raise ValueError("overflow_flag must contain exactly one element.")
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_object()
        self._router_kernel(
            topk_indices,
            topk_scores,
            overflow_flag,
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
        overflow_flag: cute.Tensor,
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
        self._overflow_flag = overflow_flag

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
        self._overflow_flag = None

    @cute.jit
    def _router_single_cta(
        self, topk_indices: cute.Tensor, topk_scores: Optional[cute.Tensor], smem_base: cute.Pointer
    ) -> None:
        if self._router_linear_cta_idx < Int32(self.router_data_cta_count):
            block_thread_count = self.router_warps_per_cta * 32
            trash_bucket = self.expert_count_padded
            histogram = self._router_smem_workspace.tensor(self.router_data_histogram_region, smem_base)
            totals = self._router_smem_workspace.tensor(self.router_data_totals_region, smem_base)
            prefix = self._router_smem_workspace.tensor(self.router_data_prefix_region, smem_base)
            warp_totals = self._router_smem_workspace.tensor(self.router_data_warp_totals_region, smem_base)
            sorted_elements = self._router_smem_workspace.tensor(self.router_data_sorted_region, smem_base)

            histogram_element_count = self.expert_count_with_trash * (self.router_warps_per_cta + 1)
            histogram_flat = cute.make_tensor(histogram.iterator, cute.make_layout((histogram_element_count,)))
            zero_round_count = ceil_div(histogram_element_count, block_thread_count)
            for zero_round in cutlass.range_constexpr(zero_round_count):
                slot = Int32(zero_round * block_thread_count) + self._router_thread_idx
                if slot < Int32(histogram_element_count):
                    histogram_flat[slot] = Int32(0)

            iket.range_push("router.histogram")
            expert_registers, score_registers = self._load_router_inputs(topk_indices, topk_scores)
            cute.arch.sync_threads()
            match_masks = self._build_histogram(expert_registers, histogram)
            self._prefix_warp_histogram(histogram, totals)
            iket.range_pop()

            iket.range_push("router.prefix_and_publish")
            publish_sizes = self._broadcast_sizes_to_peers(
                cute.make_tensor(totals.iterator, cute.make_layout((self.expert_count_padded,)))
            )
            total_valid_routes = smem_exclusive_prefix(
                cute.make_tensor(totals.iterator, cute.make_layout((self.expert_count_padded,))),
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
                expert_registers, match_masks, score_registers, sorted_elements, prefix, histogram, topk_indices.dtype
            )
            cute.arch.sync_threads()
            iket.range_pop()
            iket.range_push("router.write_out")
            self._dump_contiguous_router_output(sorted_elements, total_valid_routes)
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.sorted_metadata_ready_region), Int32(1), sem="release", scope="gpu"
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
            histogram = self._router_smem_workspace.tensor(self.router_data_histogram_region, smem_base)
            totals = self._router_smem_workspace.tensor(self.router_data_totals_region, smem_base)
            prefix = self._router_smem_workspace.tensor(self.router_data_prefix_region, smem_base)
            warp_totals = self._router_smem_workspace.tensor(self.router_data_warp_totals_region, smem_base)
            sorted_elements = self._router_smem_workspace.tensor(self.router_data_sorted_region, smem_base)
            dump_base = self._router_smem_workspace.tensor(self.router_data_base_region, smem_base)

            histogram_element_count = self.expert_count_with_trash * (self.router_warps_per_cta + 1)
            histogram_flat = cute.make_tensor(histogram.iterator, cute.make_layout((histogram_element_count,)))
            zero_round_count = ceil_div(histogram_element_count, block_thread_count)
            for zero_round in cutlass.range_constexpr(zero_round_count):
                slot = Int32(zero_round * block_thread_count) + self._router_thread_idx
                if slot < Int32(histogram_element_count):
                    histogram_flat[slot] = Int32(0)

            iket.range_push("router.histogram")
            expert_registers, score_registers = self._load_router_inputs(topk_indices, topk_scores)
            cute.arch.sync_threads()
            match_masks = self._build_histogram(expert_registers, histogram)
            self._prefix_warp_histogram(histogram, totals)
            iket.range_pop()

            iket.range_push("router.reserve_and_prefix")
            cta_histograms = self._device_workspace.tensor(self.router_cta_histograms_region)
            for expert_round in cutlass.range_constexpr(expert_round_count):
                expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_padded):
                    cta_histograms[self._router_linear_cta_idx, expert] = totals[expert]
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.router_histogram_done_region), Int32(1), sem="release", scope="gpu"
                )

            total_valid_routes = smem_exclusive_prefix(
                cute.make_tensor(totals.iterator, cute.make_layout((self.expert_count_padded,))),
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
                expert_registers, match_masks, score_registers, sorted_elements, prefix, histogram, topk_indices.dtype
            )
            cute.arch.sync_threads()
            iket.range_pop()

            iket.range_push("router.wait_source_base")
            source_base_ready = self._device_workspace.ptr(self.source_base_ready_region)
            if self._router_thread_idx == Int32(0):
                while cute.arch.load(source_base_ready, Int32, sem="acquire", scope="gpu") != Int32(1):
                    nanosleep(150)
            cute.arch.sync_threads()
            iket.range_pop()

            iket.range_push("router.write_out")
            source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
            cta_histograms = self._device_workspace.tensor(self.router_cta_histograms_region)
            for expert_round in cutlass.range_constexpr(expert_round_count):
                expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_padded):
                    dump_base[expert] = source_expert_base[expert] + cta_histograms[self._router_linear_cta_idx, expert]
            cute.arch.sync_threads()
            self._dump_router_output_by_expert(totals, prefix, dump_base, sorted_elements)
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.sorted_metadata_ready_region), Int32(1), sem="release", scope="gpu"
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
        warp_totals = self._router_smem_workspace.tensor(self.router_helper_warp_totals_region, smem_base)
        cta_histograms = self._device_workspace.tensor(self.router_cta_histograms_region)
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
                running = Int32(0)
                for cta_idx in cutlass.range_constexpr(self.router_data_cta_count):
                    count = cta_histograms[cta_idx, expert]
                    cta_histograms[cta_idx, expert] = running
                    running = running + count
                totals[expert] = running
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
                self._device_workspace.ptr(self.source_base_ready_region), Int32(1), sem="release", scope="gpu"
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

        global_expert = self._router_linear_cta_idx * Int32(self.router_warps_per_cta) + self._router_warp_idx
        if global_expert < Int32(self.expert_count):
            sizes_by_rank = self._device_workspace.tensor(self.sizes_by_rank_region)
            source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
            push_destination_base = self._device_workspace.tensor(self.push_destination_base_region)
            route_count = sizes_by_rank[self._router_local_rank, global_expert]
            source_begin = source_expert_base[global_expert]
            destination_begin = push_destination_base[global_expert]
            destination_rank = global_expert // Int32(self.experts_per_rank)
            if cutlass.const_expr(self.interleave_source_ranks):
                source_counts = [sizes_by_rank[r, global_expert] for r in range(self.world_size)]
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
            if cutlass.const_expr(self.input_dedup_mode != "off"):
                destination_routes_address = self._device_workspace.ptr(self.input_dedup_routes_region).toint() + peer_offset
                local_expert = global_expert % Int32(self.experts_per_rank)
                global_sizes = self._device_workspace.tensor(self.sizes_region)
                pool_begin = Int32(0)
                sf_begin = Int32(0)
                ready_begin = Int32(0)
                for earlier in cutlass.range_constexpr(self.experts_per_rank):
                    if Int32(earlier) < local_expert:
                        count = global_sizes[destination_rank * Int32(self.experts_per_rank) + Int32(earlier)]
                        pool_begin = pool_begin + (
                            (count + Int32(self.token_padding_block - 1)) // Int32(self.token_padding_block)
                        ) * Int32(self.token_padding_block)
                        sf_begin = sf_begin + (
                            (count + Int32(self.sf_padding_block - 1)) // Int32(self.sf_padding_block)
                        ) * Int32(self.sf_padding_block)
                        ready_begin = ready_begin + (
                            (count + Int32(self.tokens_per_fc1_ready_slot - 1))
                            // Int32(self.tokens_per_fc1_ready_slot)
                        )
            route_round_count = (route_count + Int32(31)) // Int32(32)
            for route_round in cutlass.range(route_round_count, unroll=1):
                route = Int32(route_round) * Int32(32) + self._router_lane_idx
                if route < route_count:
                    source_position = source_begin + route
                    destination_position = destination_begin + route
                    if cutlass.const_expr(self.interleave_source_ranks):
                        # Visit source ranks in order, skipping exhausted ranks.
                        # This is also the canonical row order of fc1_preact:
                        # FWD and BWD must select the same ordering.
                        round_robin_offset = Int32(0)
                        for source_rank in cutlass.range_constexpr(self.world_size):
                            count = source_counts[source_rank]
                            round_robin_offset = round_robin_offset + cutlass.min(route, count)
                            if Int32(source_rank) < Int32(self._router_local_rank) and route < count:
                                round_robin_offset = round_robin_offset + Int32(1)
                        destination_position = destination_begin + round_robin_offset
                    physical_store = Int32(destination_position < Int32(self.worst_case_token_count))
                    metadata = cute.arch.load(source_metadata + source_position, cutlass.Int64)
                    stg_b64(
                        destination_metadata_address + Int64(destination_position) * Int64(TokenSrcMetadata.nbytes),
                        metadata,
                        physical_store,
                    )
                    if cutlass.const_expr(self.input_dedup_mode != "off"):
                        # Decode through the protocol class: the record layout changed (token:u16 | topk:u8 |
                        # rank:u8 | order:u8 | cnt:u8) and a hand-rolled mask here silently indexed the wrong route.
                        decoded = TokenSrcMetadata.unpack(Int64(metadata))
                        source_token = decoded.src_token
                        source_slot = decoded.src_topk
                        route_index = (self._router_local_rank * Int32(self.max_tokens_per_rank) + source_token) * Int32(self.topk) + source_slot
                        token_in_expert = destination_position - pool_begin
                        sf_position = sf_begin + token_in_expert
                        ready_slot = ready_begin + token_in_expert // Int32(self.tokens_per_fc1_ready_slot)
                        # Zero denotes an absent route; the destination row
                        # also gives each source token/receiver a unique owner.
                        packed_route = (
                            (Int64(ready_slot) << Int64(40)) | (Int64(sf_position) << Int64(20))
                            | Int64(destination_position + Int32(1))
                        )
                        stg_b64(destination_routes_address + Int64(route_index) * Int64(8), packed_route, physical_store)
                    if cutlass.const_expr(self.apply_topk_at_fc1):
                        score = cute.arch.load(source_scores + source_position, cutlass.Float32)
                        stg_f32(
                            destination_scores_address + Int64(destination_position) * Int64(4), score, physical_store
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
                    self._peer_rank_ptr_mapper.map(metadata_ready_address, destination_rank, Int64(0)), Int32(1)
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
                    + Int64(Int32(self._router_local_rank) * Int32(self.expert_count_padded)) * Int64(4)
                )
                destination_row = cute.make_ptr(
                    cutlass.Int32, destination_row_address, AddressSpace.gmem, assumed_align=16
                )
                destination_total = cute.make_ptr(
                    cutlass.Int32, total_address + peer_offset, AddressSpace.gmem, assumed_align=16
                )
                with cute.arch.elect_one():
                    cp_async_bulk_s2g(destination_row, smem_expert_counts.iterator, row_bytes)
                    cp_reduce_async_bulk_add_u32_s2g(destination_total, smem_expert_counts.iterator, row_bytes)
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
                        self._peer_rank_ptr_mapper.map(ready_address, destination_rank, Int64(0)), Int32(1)
                    )

        return finalize

    @cute.jit
    def _compute_push_tables(self, smem_base: cute.Pointer) -> None:
        block_thread_count = self.router_warps_per_cta * 32
        owner_expert_begin = Int32(self._router_local_rank) * Int32(self.experts_per_rank)
        matrix_bytes = self.world_size * self.expert_count_padded * 4

        size_matrix = self._router_smem_workspace.tensor(self.router_helper_size_matrix_region, smem_base)
        padded_totals = self._router_smem_workspace.tensor(self.router_helper_totals_region, smem_base)
        prefix = self._router_smem_workspace.tensor(self.router_helper_prefix_region, smem_base)
        warp_totals = self._router_smem_workspace.tensor(self.router_helper_warp_totals_region, smem_base)
        load_mbarrier = self._router_smem_workspace.ptr(self.router_helper_load_mbarrier_region, smem_base)
        sizes = self._device_workspace.tensor(self.sizes_region)

        if self._router_thread_idx == Int32(0):
            cute.arch.mbarrier_init(load_mbarrier, 1)

        sizes_ready = self._device_workspace.ptr(self.sizes_ready_region)
        iket.range_push("router.wait_sizes_ready")
        if self._router_thread_idx == Int32(0):
            while cute.arch.load(sizes_ready, Int32, sem="acquire", scope="sys") != Int32(self.raw_sizes_ready_target):
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
                    (expert_size + Int32(self.token_padding_block - 1)) // Int32(self.token_padding_block)
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
                pool_expert_base[local_expert] = prefix[owner_expert_begin + local_expert] - prefix[owner_expert_begin]

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
                source_ring_offset = Int32(0)
                if cutlass.const_expr(not self.interleave_source_ranks):
                    local_ring_position = (
                        Int32(self._router_local_rank) - destination_rank + Int32(self.world_size)
                    ) % Int32(self.world_size)
                    for ring_position in cutlass.range_constexpr(self.world_size):
                        source_rank = (destination_rank + Int32(ring_position)) % Int32(self.world_size)
                        if Int32(ring_position) < local_ring_position:
                            source_ring_offset = source_ring_offset + size_matrix[source_rank, global_expert]
                push_destination_base[global_expert] = destination_pool_base + source_ring_offset
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("router.apply_receive_limit")
        if self._router_thread_idx == Int32(0):
            warp_totals[0] = Int32(0)
            warp_totals[1] = Int32(0)
        cute.arch.sync_threads()

        # sizes[] contains the same pre-truncation, all-source expert totals on
        # every rank once raw_sizes_ready_target has been observed. Compute the
        # group-wide policy bit locally so no post-kernel collective is needed.
        # Slot 0 counts overflowing destinations; slot 1 records this rank's
        # destination bit for local-only truncation.
        rank_round_count = ceil_div(self.world_size, block_thread_count)
        for rank_round in cutlass.range_constexpr(rank_round_count):
            destination_rank = Int32(rank_round * block_thread_count) + self._router_thread_idx
            if destination_rank < Int32(self.world_size):
                destination_expert_begin = destination_rank * Int32(self.experts_per_rank)
                raw_destination_total = Int32(0)
                for local_expert in cutlass.range_constexpr(self.experts_per_rank):
                    raw_destination_total = (
                        raw_destination_total
                        + sizes[destination_expert_begin + Int32(local_expert)]
                    )
                destination_overflow = Int32(
                    raw_destination_total > Int32(self.max_recv_size_per_rank)
                )
                if destination_overflow:
                    cute.arch.atomic_add(
                        warp_totals.iterator,
                        Int32(1),
                        scope="cta",
                    )
                if destination_rank == Int32(self._router_local_rank):
                    warp_totals[1] = destination_overflow
        cute.arch.sync_threads()
        global_did_overflow = Int32(warp_totals[0] > Int32(0))
        local_did_overflow = warp_totals[1]

        if cutlass.const_expr(self.drop_on_overflow):
            local_expert_rounds = ceil_div(self.experts_per_rank, block_thread_count)
            for expert_round in cutlass.range_constexpr(local_expert_rounds):
                local_expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
                if local_expert < Int32(self.experts_per_rank):
                    # Keep every raw row that still lands inside the statically allocated padded pool.
                    raw_size = sizes[owner_expert_begin + local_expert]
                    retained_size = Int32(0)
                    remaining_capacity = Int32(self.worst_case_token_count) - pool_expert_base[local_expert]
                    if remaining_capacity > Int32(0):
                        retained_size = cutlass.min(raw_size, remaining_capacity)
                    if local_did_overflow:
                        sizes[owner_expert_begin + local_expert] = retained_size

        if self._router_thread_idx == Int32(0):
            cute.arch.store(
                self._overflow_flag.iterator,
                global_did_overflow,
                sem="relaxed",
                scope="sys",
            )
        cute.arch.sync_threads()

        if cutlass.const_expr(self.drop_on_overflow):
            self._publish_push_table_and_sizes_ready(sizes_ready)
        else:
            if global_did_overflow:
                if self._router_thread_idx == Int32(0):
                    cute.arch.fence_acq_rel_sys()
                    _device_trap()
            else:
                self._publish_push_table_and_sizes_ready(sizes_ready)
        iket.range_pop()

    @cute.jit
    def _publish_push_table_and_sizes_ready(self, sizes_ready: cute.Pointer) -> None:
        if self._router_thread_idx == Int32(0):
            cute.arch.fence_acq_rel_gpu()
            cute.arch.atomic_add(
                self._device_workspace.ptr(self.push_table_ready_region), Int32(1), sem="relaxed", scope="gpu"
            )
            cute.arch.atomic_add(sizes_ready, Int32(1), sem="relaxed", scope="gpu")

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
        topk_vectors = cute.logical_divide(cute.zipped_divide(topk_flat, (tile_span,)), (elements_per_vector, None))
        load_atom = _copy_atom(topk_indices.dtype, 128)
        expert_registers = cute.make_rmem_tensor((elements_per_thread,), cutlass.Int32)
        if cutlass.const_expr(topk_indices.dtype.width == 64):
            raw_indices = cute.make_rmem_tensor((elements_per_thread,), topk_indices.dtype)
            raw_vectors = cute.zipped_divide(raw_indices, (elements_per_vector,))
            for load_round in cutlass.range_constexpr(load_round_count):
                tile_begin = Int32(load_round * tile_span) + self._router_grid_thread_idx * Int32(elements_per_vector)
                if tile_begin < actual_elements:
                    cute.copy(
                        load_atom,
                        _mark_alignment(topk_vectors[(None, self._router_grid_thread_idx), load_round], 16),
                        raw_vectors[None, load_round],
                    )
        else:
            expert_vectors = cute.zipped_divide(expert_registers, (elements_per_vector,))
            for load_round in cutlass.range_constexpr(load_round_count):
                tile_begin = Int32(load_round * tile_span) + self._router_grid_thread_idx * Int32(elements_per_vector)
                if tile_begin < actual_elements:
                    cute.copy(
                        load_atom,
                        _mark_alignment(topk_vectors[(None, self._router_grid_thread_idx), load_round], 16),
                        expert_vectors[None, load_round],
                    )

        score_registers = None
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_registers = cute.make_rmem_tensor((elements_per_thread,), cutlass.Float32)
            scores_flat = cute.make_tensor(topk_scores.iterator, cute.make_layout((maximum_elements,)))
            score_vectors = cute.logical_divide(
                cute.zipped_divide(scores_flat, (tile_span,)), (elements_per_vector, None)
            )
            score_atom = _copy_atom(cutlass.Float32, elements_per_vector * 32)
            score_register_vectors = cute.zipped_divide(score_registers, (elements_per_vector,))
            for load_round in cutlass.range_constexpr(load_round_count):
                tile_begin = Int32(load_round * tile_span) + self._router_grid_thread_idx * Int32(elements_per_vector)
                if tile_begin < actual_elements:
                    cute.copy(
                        score_atom,
                        _mark_alignment(score_vectors[(None, self._router_grid_thread_idx), load_round], 16),
                        score_register_vectors[None, load_round],
                    )

        expert_registers_u32 = cute.recast_tensor(expert_registers, cutlass.Uint32)
        if cutlass.const_expr(topk_indices.dtype.width == 64):
            raw_indices_i32 = cute.recast_tensor(raw_indices, cutlass.Int32)
        for register_idx in cutlass.range_constexpr(elements_per_thread):
            if cutlass.const_expr(topk_indices.dtype.width == 64):
                expert_registers[register_idx] = raw_indices_i32[2 * register_idx]
            token_idx, _ = self._router_value_coordinate(register_idx, topk_indices.dtype)
            is_invalid = (expert_registers_u32[register_idx] >= cutlass.Uint32(self.expert_count)) | (
                token_idx >= actual_token_count
            )
            if is_invalid:
                expert_registers[register_idx] = Int32(self.expert_count_padded)
        return expert_registers, score_registers

    @cute.jit
    def _build_histogram(self, expert_registers: cute.Tensor, histogram: cute.Tensor) -> cute.Tensor:
        register_count = cute.size(expert_registers)
        match_masks = cute.make_rmem_tensor((register_count,), cutlass.Int32)
        lane_mask_less_than = (Int32(1) << self._router_lane_idx) - Int32(1)
        for register_idx in cutlass.range_constexpr(register_count):
            expert = expert_registers[register_idx]
            match_mask = Int32(cute.arch.match_sync(0xFFFFFFFF, expert, kind="any"))
            match_masks[register_idx] = match_mask
            histogram_slot = expert * Int32(self.router_warps_per_cta + 1) + self._router_warp_idx
            rank_in_group = Int32(cute.arch.popc(match_mask & lane_mask_less_than))
            if rank_in_group == Int32(0):
                cute.arch.atomic_add(
                    histogram.iterator + histogram_slot, Int32(cute.arch.popc(match_mask)), sem="relaxed", scope="cta"
                )
        cute.arch.sync_threads()
        return match_masks

    @cute.jit
    def _prefix_warp_histogram(self, histogram: cute.Tensor, totals: cute.Tensor) -> None:
        """Turn per-warp counts into fixed warp bases and expert totals."""
        block_thread_count = self.router_warps_per_cta * 32
        expert_round_count = ceil_div(self.expert_count_with_trash, block_thread_count)
        for expert_round in cutlass.range_constexpr(expert_round_count):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if expert < Int32(self.expert_count_with_trash):
                running = Int32(0)
                for warp in cutlass.range_constexpr(self.router_warps_per_cta):
                    count = histogram[expert, warp]
                    histogram[expert, warp] = running
                    running = running + count
                totals[expert] = running
        cute.arch.sync_threads()

    @cute.jit
    def _sort_router_elements(
        self,
        expert_registers: cute.Tensor,
        match_masks: cute.Tensor,
        score_registers: Optional[cute.Tensor],
        sorted_elements: cute.Tensor,
        expert_run_starts: cute.Tensor,
        warp_histogram: cute.Tensor,
        topk_index_type: type,
    ) -> None:
        """Stable CTA-local counting-sort scatter.

        The cursor for an expert is private to one warp.  ``match.any`` gives
        every equal-expert lane a fixed rank, and only that group's first lane
        advances the cursor.  Therefore the output order is a pure function of
        warp id, register round, and lane id rather than atomic arrival order.
        """
        register_count = cute.size(expert_registers)
        lane_mask_less_than = (Int32(1) << self._router_lane_idx) - Int32(1)
        for register_idx in cutlass.range_constexpr(register_count):
            expert = expert_registers[register_idx]
            match_mask = match_masks[register_idx]
            group_size = Int32(cute.arch.popc(match_mask))
            rank_in_group = Int32(cute.arch.popc(match_mask & lane_mask_less_than))
            warp_base = warp_histogram[expert, self._router_warp_idx]
            token_idx, topk_slot = self._router_value_coordinate(register_idx, topk_index_type)
            flat_topk_index = token_idx * Int32(self.topk) + topk_slot
            destination = expert_run_starts[expert] + warp_base + rank_in_group

            # Every lane in an equal-expert group reads the old cursor before
            # its leader advances it for the next register round.
            cute.arch.sync_warp()
            if rank_in_group == Int32(0):
                warp_histogram[expert, self._router_warp_idx] = warp_base + group_size
            if cutlass.const_expr(self.apply_topk_at_fc1):
                sorted_elements[destination] = _SortedElement(flat_topk_index, score_registers[register_idx]).pack()
            else:
                sorted_elements[destination] = _SortedElement(flat_topk_index, None).pack()
            cute.arch.sync_warp()

    @cute.jit
    def _router_value_coordinate(self, register_idx: int, topk_index_type: type) -> Tuple[Int32, Int32]:
        elements_per_vector = 128 // topk_index_type.width
        tile_span = elements_per_vector * self.router_data_cta_count * self.router_warps_per_cta * 32
        flat_index = Int32(
            register_idx // elements_per_vector * tile_span + register_idx % elements_per_vector
        ) + self._router_grid_thread_idx * Int32(elements_per_vector)
        return (flat_index // Int32(self.topk), flat_index % Int32(self.topk))

    @cute.jit
    def _dump_contiguous_router_output(self, sorted_elements: cute.Tensor, total_valid_routes: Int32) -> None:
        block_thread_count = self.router_warps_per_cta * 32
        metadata_address = self._device_workspace.ptr(self.sorted_metadata_region).toint()
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_address = self._device_workspace.ptr(self.sorted_scores_region).toint()
        dump_round_count = (total_valid_routes + Int32(block_thread_count - 1)) // Int32(block_thread_count)
        for dump_round in cutlass.range(dump_round_count, unroll=4):
            position = Int32(dump_round * block_thread_count) + self._router_thread_idx
            predicate = Int32(position < total_valid_routes)
            element = _SortedElement.from_packed(sorted_elements[position])
            metadata = TokenSrcMetadata(
                src_rank=Int32(self._router_local_rank),
                src_token=(element.flat_topk_index // Int32(self.topk)),
                src_topk=(element.flat_topk_index % Int32(self.topk)),
            )
            stg_b64(metadata_address + Int64(position) * Int64(TokenSrcMetadata.nbytes), metadata.pack(), predicate)
            if cutlass.const_expr(self.apply_topk_at_fc1):
                stg_f32(score_address + Int64(position) * Int64(4), element.topk_score, predicate)

    @cute.jit
    def _dump_router_output_by_expert(
        self,
        expert_totals: cute.Tensor,
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
                run_length = expert_totals[expert]
                dump_begin = expert_dump_bases[expert]
                route_round_count = (run_length + Int32(31)) // Int32(32)
                for route_round in cutlass.range(route_round_count, unroll=1):
                    route = Int32(route_round) * Int32(32) + self._router_lane_idx
                    predicate = Int32(route < run_length)
                    element = _SortedElement.from_packed(sorted_elements[predicate * (run_begin + route)])
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
                        stg_f32(score_address + Int64(output_position) * Int64(4), element.topk_score, predicate)

    @cute.jit
    def local_expert_sizes(self, device_workspace: DeviceWorkspace, local_rank: Int32) -> cute.Tensor:
        """Return this rank's contiguous expert-size view."""
        sizes = device_workspace.tensor(self.sizes_region)
        expert_begin = local_rank * Int32(self.experts_per_rank)
        return cute.make_tensor(sizes.iterator + expert_begin, cute.make_layout((self.experts_per_rank,)))

    @property
    def metadata_ready_target(self) -> int:
        return self.router_push_cta_count * self.world_size

    @property
    def raw_sizes_ready_target(self) -> int:
        return self.world_size

    @property
    def published_sizes_ready_target(self) -> int:
        return self.world_size + 1

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
    def wait_for_sizes_ready(self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        lane_idx = thread_idx % Int32(32)
        if lane_idx == Int32(0):
            sizes_ready = device_workspace.ptr(self.sizes_ready_region)
            while cute.arch.load(sizes_ready, Int32, sem="acquire", scope="gpu") != Int32(
                self.published_sizes_ready_target
            ):
                nanosleep(sleep_cycles)
        cute.arch.sync_warp()

    @cute.jit
    def wait_for_metadata_ready(self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000) -> None:
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


class TokenCommDeterministic(KernelComponent):
    """Fused communication with a fixed logical token-position sequence."""

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
    token_back_schedule_region = "nvlink.token_comm.token_back_schedule"  # nosec B105

    token_in_mbarrier_region = "nvlink.token_comm.main_smem.token_in_mbarriers"  # nosec B105
    token_back_mbarrier_region = "nvlink.token_comm.main_smem.token_back_mbarriers"  # nosec B105
    expert_sizes_smem_region = "nvlink.token_comm.main_smem.expert_sizes"
    token_in_activation_smem_region = "nvlink.token_comm.main_smem.token_in_activation"  # nosec B105
    token_in_sf_smem_region = "nvlink.token_comm.main_smem.token_in_sf"  # nosec B105
    token_back_activation_smem_region = "nvlink.token_comm.main_smem.token_back_activation"  # nosec B105
    token_back_sf_smem_region = "nvlink.token_comm.main_smem.token_back_sf"  # nosec B105

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "world_size": int,
            "expert_count": int,
            "topk": int,
            "max_tokens_per_rank": int,
            "max_recv_size_per_rank": int,
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
            # Optional so existing component users keep the four-warp ABI.
            # Embedding kernels must still launch exactly this many transfer
            # warps and include them in their block-wide synchronization.
            "token_in_transfer_warp_count": OptionalRequirement(int),
            "token_in_flag_batch": int,
            "token_in_async_flags": OptionalRequirement(bool),
            "input_dedup_mode": OptionalRequirement(str),
            "interleave_source_ranks": OptionalRequirement(bool),
            "token_back_mode": str,
            "token_back_schedule_mode": str,
            "reduce_topk_in_kernel": bool,
            "drop_on_overflow": bool,
            "token_in_stage_split": OptionalRequirement(bool),
            "external_dispatch_pool": OptionalRequirement(bool),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.world_size = problem_desc["world_size"]
        self.expert_count = problem_desc["expert_count"]
        self.topk = problem_desc["topk"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.max_recv_size_per_rank = min(
            problem_desc["max_recv_size_per_rank"], self.world_size * self.max_tokens_per_rank * self.topk
        )
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = problem_desc["quant_kind"]
        self.combine_format = problem_desc["combine_format"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.tokens_per_fc1_ready_slot = impl_desc["tokens_per_fc1_ready_slot"]
        self.fc2_done_signals_per_token_tile = impl_desc["fc2_done_signals_per_token_tile"]
        self.promised_launchable_sm_count = impl_desc["promised_launchable_sm_count"]
        self.transfer_warp_count = impl_desc.get(
            "token_in_transfer_warp_count", type(self).transfer_warp_count
        )
        self.transfer_thread_count = self.transfer_warp_count * 32
        self.token_in_flag_batch = impl_desc["token_in_flag_batch"]
        self.token_in_async_flags = impl_desc.get("token_in_async_flags", False)
        self.token_back_mode: TokenBackMode = impl_desc["token_back_mode"]
        self.token_back_schedule_mode: TokenBackScheduleMode = impl_desc["token_back_schedule_mode"]
        self.reduce_topk_in_kernel = impl_desc["reduce_topk_in_kernel"]
        self.drop_on_overflow = impl_desc["drop_on_overflow"]
        self.token_in_stage_split = impl_desc.get("token_in_stage_split", False)
        self._token_in_stage_chunk_count = 2 if self.token_in_stage_split else 1
        self.external_dispatch_pool = impl_desc.get("external_dispatch_pool", False)

        self._validate_configuration()
        self._router = _MetadataPushRouter(problem_desc, impl_desc)
        self.interleave_source_ranks = self._router.interleave_source_ranks
        self.input_dedup_mode = self._router.input_dedup_mode
        self.input_dedup_store_mode = self._router.input_dedup_store_mode
        self.input_dedup_order = self._router.input_dedup_order
        if self.input_dedup_mode != "off" and self.quant_kind not in ("mxfp8_e4m3", "mxfp8_e5m2"):
            raise ValueError("Input dedup requires MXFP8")
        self._nvlink_barrier = NvlinkBarrier(world_size=self.world_size, barrier_id=self.grid_sync_barrier_id)
        self._device_workspace = None
        self._token_comm_args = None
        self._local_rank = None
        self._linear_cta_idx = None
        self._transfer_warp_idx = None
        self._lane_idx = None

    def _validate_configuration(self) -> None:
        positive_fields = ("hidden_size", "token_padding_block", "sf_padding_block", "tokens_per_fc1_ready_slot")
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")
        if self.quant_kind not in _quant_spec:
            raise ValueError(f"Unsupported quant_kind {self.quant_kind!r}.")
        if self.token_back_mode not in ("epi_warps", "standalone_warps", "reuse_dispatch_warps"):
            raise ValueError(f"Unsupported token_back_mode {self.token_back_mode!r}.")
        if self.token_back_schedule_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"token_back_schedule_mode must be static or atomic_counter, got {self.token_back_schedule_mode!r}."
            )
        if self.token_in_async_flags and self.token_in_flag_batch != 1:
            raise ValueError("Asynchronous token-in flags require token_in_flag_batch=1.")
        if not 1 <= self.token_in_flag_batch <= 32:
            raise ValueError(f"token_in_flag_batch must be in [1, 32], got {self.token_in_flag_batch}.")
        if (
            isinstance(self.transfer_warp_count, bool)
            or not isinstance(self.transfer_warp_count, int)
            or not 1 <= self.transfer_warp_count <= 32
        ):
            raise ValueError(
                "token_in_transfer_warp_count must be in [1, 32], got "
                f"{self.transfer_warp_count}."
            )
        if self.tokens_per_fc1_ready_slot % self.token_padding_block != 0:
            raise ValueError("tokens_per_fc1_ready_slot must be divisible by token_padding_block.")
        _pad_lo = min(self.sf_padding_block, self.token_padding_block)
        _pad_hi = max(self.sf_padding_block, self.token_padding_block)
        if _pad_hi % _pad_lo != 0:
            raise ValueError("sf_padding_block and token_padding_block must be power-of-two multiples of each other.")
        if self.token_back_enabled and self.fc2_done_signals_per_token_tile <= 0:
            raise ValueError("fc2_done_signals_per_token_tile must be positive when token-back is enabled.")
        if self.reduce_topk_in_kernel and self.combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError("In-kernel top-k reduction requires BF16 combine data.")
        if self.activation_is_quantized:
            element_block = self.activation_sf_vector_size * 4
            if self.hidden_size % element_block != 0:
                raise ValueError(f"{self.quant_kind} requires hidden_size divisible by {element_block}.")
        else:
            # The 1D bulk copies move whole token rows, so each row must be whole 16-byte lines.
            row_element_block = 128 // int(self.activation_dtype.width)
            if self.hidden_size % row_element_block != 0:
                raise ValueError(f"The BF16 wire requires hidden_size divisible by {row_element_block}.")
        if self.sf_padding_block % 128 != 0:
            raise ValueError("sf_padding_block must be a multiple of 128.")
        if self.token_in_stage_split:
            # Scale-free wire only: the SF pull and the token-back stage sizing both assume one
            # whole-row stage per transfer warp.
            if self.activation_is_quantized or self.token_back_enabled:
                raise NotImplementedError(
                    "token_in_stage_split requires the BF16 wire with token_back disabled (epi_warps)."
                )
            if self.bytes_per_token % (self._token_in_stage_chunk_count * 16) != 0:
                raise ValueError("token_in_stage_split must cut a token row into whole 16-byte chunks.")

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.world_size

    @property
    def activation_is_quantized(self) -> bool:
        return _quant_spec[self.quant_kind][1] is not None

    @property
    def activation_dtype(self) -> type:
        return _quant_spec[self.quant_kind][0]

    @property
    def activation_sf_dtype(self) -> type:
        return _quant_spec[self.quant_kind][1]

    @property
    def activation_sf_vector_size(self) -> int:
        return _quant_spec[self.quant_kind][2]

    @property
    def bytes_per_token(self) -> int:
        return self.hidden_size * int(self.activation_dtype.width) // 8

    @property
    def activation_sf_hidden_padded(self) -> int:
        if not self.activation_is_quantized:
            return 0
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
        return self.token_back_mode != "epi_warps"  # nosec B105

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
        # The SF pool has to be at least as tall as the data pool: once
        # token_padding_block may exceed sf_padding_block, an expert's data
        # cursor can walk past the SF-capacity-derived row count.
        sf_capacity = self._router.receive_capacity(self.sf_padding_block).padded_route_count
        return max(sf_capacity, self.worst_case_token_count)

    @property
    def max_fc1_ready_slot_count(self) -> int:
        return (
            self._router.receive_capacity(self.tokens_per_fc1_ready_slot).padded_route_count
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
        overflow_flag: cute.Tensor,
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
            overflow_flag,
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper_host,
            device_workspace,
            stream,
        )

    @cute.jit
    def local_expert_sizes(self, device_workspace: DeviceWorkspace, local_rank: Int32) -> cute.Tensor:
        return self._router.local_expert_sizes(device_workspace, local_rank)

    @cute.jit
    def wait_for_sizes_ready(self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000) -> None:
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
        if cutlass.const_expr(self.external_dispatch_pool and token_comm_args.fc1_activation_pool is None):
            raise ValueError("external_dispatch_pool requires TokenCommArgs.fc1_activation_pool.")
        if cutlass.const_expr(not self.external_dispatch_pool and token_comm_args.fc1_activation_pool is not None):
            raise ValueError("TokenCommArgs.fc1_activation_pool is only consumed under external_dispatch_pool.")
        self._device_workspace = device_workspace
        self._token_comm_args = token_comm_args
        self._local_rank = local_rank
        self._linear_cta_idx = linear_cta_idx
        thread_idx, _, _ = cute.arch.thread_idx()
        transfer_thread_idx = thread_idx % Int32(self.transfer_thread_count)
        self._transfer_warp_idx = cute.arch.make_warp_uniform(transfer_thread_idx // Int32(32))
        self._lane_idx = transfer_thread_idx % Int32(32)
        self._nvlink_barrier.assign_device_members(device_workspace, token_comm_args.peer_rank_ptr_mapper)

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

    def __new_from_mlir_values__(self, values: list) -> "TokenCommDeterministic":
        if values:
            raise ValueError("TokenCommDeterministic carries no MLIR values.")
        return self

    def _register_main_workspace(self, workspace: DeviceWorkspace) -> None:
        workspace.register(
            self.fc1_ready_region,
            cutlass.Int32,
            (self.max_fc1_ready_slot_count,),
            buffer_space="local",
            reset="tail_reset",
        )
        if not self.external_dispatch_pool:
            activation_element_count = self.worst_case_token_count * self.hidden_size
            workspace.register(
                self.fc1_activation_region,
                self.activation_dtype,
                (activation_element_count,),
                buffer_space="local",
                byte_alignment=128,
            )
        if self.activation_is_quantized:
            activation_sf_element_count = self.worst_case_sf_token_count * self.activation_sf_hidden_padded
            workspace.register(
                self.fc1_activation_sf_region,
                self.activation_sf_dtype,
                (activation_sf_element_count,),
                buffer_space="local",
                byte_alignment=128,
            )
        if self.token_back_enabled:
            workspace.register(
                self.fc2_done_region, cutlass.Int32, (self.experts_per_rank,), buffer_space="local", reset="tail_reset"
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
        if self.token_back_enabled and self.token_back_schedule_mode == "atomic_counter":  # nosec B105
            workspace.register(
                self.token_back_schedule_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
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
            self.expert_sizes_smem_region, cutlass.Int32, (self.experts_per_rank,), byte_alignment=16
        )
        transfer_overlay = workspace.create_overlay("nvlink.token_comm.main_smem.transfer")
        token_in_lifetime = transfer_overlay.add_lifetime("token_in")
        token_in_lifetime.register_tensor(
            self.token_in_activation_smem_region,
            self.activation_dtype,
            (self.transfer_warp_count, self.hidden_size // self._token_in_stage_chunk_count),
            byte_alignment=16,
        )
        if self.activation_is_quantized:
            token_in_lifetime.register_tensor(
                self.token_in_sf_smem_region,
                self.activation_sf_dtype,
                (self.transfer_warp_count, (self.activation_sf_vector_size, self.activation_sf_hidden_padded)),
                stride=(self.activation_sf_hidden_padded, (0, 1)),
                byte_alignment=16,
            )
        if not self.token_back_enabled:
            return

        if self.token_back_mode == "standalone_warps":  # nosec B105
            token_back_lifetime = workspace.create_overlay(
                "nvlink.token_comm.main_smem.standalone_token_back"
            ).add_lifetime("token_back")
        else:
            token_back_lifetime = transfer_overlay.add_lifetime("token_back")

        if self.token_back_mode == "standalone_warps":  # nosec B105
            available_bytes_per_warp = self.standalone_chunk_bytes
        else:
            activation_bytes = self.bytes_per_token
            sf_bytes = (
                self.activation_sf_hidden_padded * int(self.activation_sf_dtype.width) // 8
                if self.activation_is_quantized
                else 0
            )
            available_bytes_per_warp = activation_bytes + sf_bytes

        if self.token_back_push_data:
            bytes_per_output_token = self.hidden_size * int(self.combine_format.act_dtype.width) // 8
            if self.token_back_mode == "standalone_warps":  # nosec B105
                chunk_bytes = self.standalone_chunk_bytes
            elif available_bytes_per_warp < bytes_per_output_token:
                chunk_bytes = self.bytes_per_token
            else:
                chunk_bytes = bytes_per_output_token
            if self.token_back_mode != "standalone_warps" and bytes_per_output_token % chunk_bytes != 0:  # nosec B105
                raise ValueError("Token-back data chunk bytes must divide one row.")
            chunk_elements = chunk_bytes * 8 // int(self.combine_format.act_dtype.width)
            token_back_lifetime.register_tensor(
                self.token_back_activation_smem_region,
                self.combine_format.act_dtype,
                (self.transfer_warp_count, chunk_elements),
                byte_alignment=16,
            )
        if self.token_back_push_sf:
            sf_row_bytes = self.combine_sf_hidden_padded * int(self.combine_format.scale_dtype.width) // 8
            if sf_row_bytes > available_bytes_per_warp:
                raise ValueError("Token-back scale row exceeds its per-warp stage.")
            token_back_lifetime.register_tensor(
                self.token_back_sf_smem_region,
                self.combine_format.scale_dtype,
                (self.transfer_warp_count, (self.combine_format.scale_block, self.combine_sf_hidden_padded)),
                stride=(self.combine_sf_hidden_padded, (0, 1)),
                byte_alignment=16,
            )

    @cute.jit
    def fc1_ready_counter_pointer(self, device_workspace: DeviceWorkspace) -> cute.Pointer:
        return device_workspace.ptr(self.fc1_ready_region)

    @property
    def dispatch_pool_shape(self) -> Tuple[int, int]:
        """Row-major ``(pool_capacity, hidden)`` geometry of the dispatch pool, workspace-owned or external."""
        return (self.worst_case_token_count, self.hidden_size)

    @cute.jit
    def fc1_activation_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        if cutlass.const_expr(self.external_dispatch_pool):
            raise ValueError("The dispatch pool is caller-owned under external_dispatch_pool; use the passed tensor.")
        return cute.make_tensor(
            device_workspace.ptr(self.fc1_activation_region),
            cute.make_layout(self.dispatch_pool_shape, stride=(self.hidden_size, 1)),
        )

    @cute.jit
    def fc1_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.activation_is_quantized):
            return None
        layout = tile_atom_to_shape_SF(
            (self.worst_case_sf_token_count, self.hidden_size, 1), self.activation_sf_vector_size
        )
        return cute.make_tensor(device_workspace.ptr(self.fc1_activation_sf_region), cute.select(layout, mode=[0, 1]))

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
                (self.worst_case_token_count, 1, self.hidden_size), stride=(self.hidden_size, self.hidden_size, 1)
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
                    (self.combine_format.scale_block, self.hidden_size // self.combine_format.scale_block),
                ),
                stride=(self.combine_sf_hidden_padded, self.combine_sf_hidden_padded, (0, 1)),
            ),
        )

    @cute.jit
    def pre_reduced_activation_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        """Return the source-domain combine plane when top-k reduction is separate."""
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            return None
        return device_workspace.tensor(self.pre_reduced_activation_region)

    @cute.jit
    def pre_reduced_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        """Return the source-domain combine scale plane when one is required."""
        if cutlass.const_expr(self.reduce_topk_in_kernel or not self.combine_format.is_quantized):
            return None
        return device_workspace.tensor(self.pre_reduced_activation_sf_region)

    @cute.jit
    def _load_input_dedup_route_batch(
        self, routes: cute.Pointer, source_global_token: Int32, route_batch: Int32,
    ) -> Int64:
        """Keep one route per lane; predicate the final partial batch."""
        route_slot = route_batch * Int32(32) + self._lane_idx
        record = Int64(0)
        if route_slot < Int32(self.topk):
            record = cute.arch.load(
                routes + source_global_token * Int32(self.topk) + route_slot, Int64
            )
        return record

    @cute.jit
    def _token_in_deduplicated(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """One remote TMA payload per source token/receiver group, original row outputs.

        Router writes a direct slot index before its original metadata publication.
        Each slot has one source-router writer. Full tail drain precedes index reset.
        The original resident communication warps guarantee progress; no new kernel,
        cross-CTA waiting, mathematical operation, or reduced output is introduced.
        """
        iket.range_push("token_in.deduplicated_payload")
        lane_idx = self._lane_idx
        transfer_warp_idx = self._transfer_warp_idx
        route_batch_count = ceil_div(self.topk, 32)
        global_warp_idx = self._linear_cta_idx * Int32(self.transfer_warp_count) + transfer_warp_idx
        global_warp_count = Int32(self.promised_launchable_sm_count * self.transfer_warp_count)
        routes = self._device_workspace.ptr(self._router.input_dedup_routes_region)
        stage = smem_workspace.tensor(self.token_in_activation_smem_region, smem_base)[transfer_warp_idx, None]
        sf_stage = smem_workspace.tensor(self.token_in_sf_smem_region, smem_base)[transfer_warp_idx, (None, None)]
        barrier = smem_workspace.ptr(self.token_in_mbarrier_region, smem_base) + transfer_warp_idx
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(barrier, 1)
        cute.arch.sync_warp()
        activation_bytes = cute.cosize(stage) * int(self.activation_dtype.width) // 8
        sf_bytes = cute.cosize(sf_stage) * int(self.activation_sf_dtype.width) // 8
        source_sf_values = cute.slice_(sf_stage, (0, None))
        source_sf_vectors = cute.zipped_divide(source_sf_values, (4,))
        sf_atom = _copy_atom(self.activation_sf_dtype, 4 * int(self.activation_sf_dtype.width))
        destination_sf = self.fc1_activation_sf_tensor(self._device_workspace)
        if cutlass.const_expr(self.external_dispatch_pool):
            destination_data = self._token_comm_args.fc1_activation_pool.iterator
        else:
            destination_data = self._device_workspace.ptr(self.fc1_activation_region)
        ready = self._device_workspace.ptr(self.fc1_ready_region)
        flags = make_flag_batch_tracker(use_async=self.token_in_async_flags, flag_address=Int64(0),
                    accumulated_flags=Int32(0), phase=Int32(0), thread_idx=lane_idx)
        pull_phase = Int32(0)
        # Traverse the same dense, expert-major valid rows as original
        # token_in; physical pool padding never becomes a work item.
        owned_sizes = smem_workspace.tensor(self.expert_sizes_smem_region, smem_base)
        pool_bases = self._router.pool_expert_base_tensor(self._device_workspace)
        metadata_pointer = self._router.token_src_metadata_pointer(self._device_workspace)
        work_limit = Int32(0)
        for expert_sum in cutlass.range_constexpr(self.experts_per_rank):
            work_limit = work_limit + owned_sizes[Int32(expert_sum)]
        current_expert = Int32(0)
        expert_valid_begin = Int32(0)
        expert_valid_end = owned_sizes[Int32(0)]
        work = global_warp_idx
        while work < work_limit:
            # work < sum(sizes) proves a nonempty expert remains, including
            # leading/interior zero-size experts and the final partial wave.
            while work >= expert_valid_end:
                expert_valid_begin = expert_valid_end
                current_expert = current_expert + Int32(1)
                expert_valid_end = expert_valid_end + owned_sizes[current_expert]
            pool_row = pool_bases[current_expert] + work - expert_valid_begin
            source_rank_from_row = Int32(0)
            source_token_from_row = Int32(0)
            if lane_idx == Int32(0):
                metadata = TokenSrcMetadata.load(
                    metadata_pointer.toint() + Int64(pool_row) * Int64(TokenSrcMetadata.nbytes)
                )
                source_rank_from_row = metadata.src_rank
                source_token_from_row = metadata.src_token
            source_rank_from_row = Int32(cute.arch.shuffle_sync(source_rank_from_row, Int32(0)))
            source_token_from_row = Int32(cute.arch.shuffle_sync(source_token_from_row, Int32(0)))
            source_global_token = source_rank_from_row * Int32(self.max_tokens_per_rank) + source_token_from_row
            # Every route owns a distinct destination row. Select the smallest
            # row across ALL batches, preserving expert-major ownership without
            # packing a local expert ID. Repeated experts still have distinct rows.
            record = self._load_input_dedup_route_batch(routes, source_global_token, Int32(0))
            route_row = Int32(record & Int64(0xFFFFF))
            earlier_owner = (route_row != Int32(0)) & (route_row < pool_row + Int32(1))
            has_earlier_owner = cute.arch.vote_ballot_sync(earlier_owner)
            if cutlass.const_expr(self.topk > 32):
                for route_batch in cutlass.range(1, route_batch_count, unroll=1):
                    other = self._load_input_dedup_route_batch(routes, source_global_token, route_batch)
                    other_row = Int32(other & Int64(0xFFFFF))
                    earlier_owner = (other_row != Int32(0)) & (other_row < pool_row + Int32(1))
                    has_earlier_owner = has_earlier_owner | cute.arch.vote_ballot_sync(earlier_owner)
            # The current valid row guarantees at least one local route, even
            # when this receiver's first matching route is in a later batch.
            if has_earlier_owner == Int32(0):
                source_rank = source_global_token // Int32(self.max_tokens_per_rank)
                source_token = source_global_token % Int32(self.max_tokens_per_rank)
                peer_offset = self._token_comm_args.peer_rank_ptr_mapper.map(Int64(0), source_rank, Int64(0))
                source_data_address = self._token_comm_args.activation.iterator.toint() + peer_offset + Int64(source_token) * Int64(self.bytes_per_token)
                remote_sf = cute.make_tensor(cute.make_ptr(
                    self._token_comm_args.activation_sf.dtype,
                    self._token_comm_args.activation_sf.iterator.toint() + peer_offset,
                    AddressSpace.gmem, assumed_align=self._token_comm_args.activation_sf.iterator.max_alignment),
                    self._token_comm_args.activation_sf.layout)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(barrier, Int32(activation_bytes + sf_bytes))
                    tma_load_1d(stage.iterator, source_data_address, barrier, Int32(activation_bytes))
                    tma_load_1d(sf_stage.iterator, remote_sf[Int64(source_token), None].iterator, barrier, Int32(sf_bytes))
                cute.arch.sync_warp()
                cute.arch.mbarrier_wait(barrier, pull_phase)
                for route_batch in cutlass.range(route_batch_count, unroll=1):
                    batch_record = record
                    if cutlass.const_expr(self.topk > 32):
                        if route_batch != Int32(0):
                            batch_record = self._load_input_dedup_route_batch(
                                routes, source_global_token, route_batch
                            )
                    for slot in cutlass.range_constexpr(min(self.topk, 32)):
                        lo = cute.arch.shuffle_sync(Int32(batch_record & Int64(0xFFFFFFFF)), Int32(slot))
                        hi = cute.arch.shuffle_sync(Int32(batch_record >> Int64(32)), Int32(slot))
                        route = (Int64(hi) << Int64(32)) | (Int64(lo) & Int64(0xFFFFFFFF))
                        if route != Int64(0):
                            destination_row = Int32(route & Int64(0xFFFFF)) - Int32(1)
                            sf_row = Int32((route >> Int64(20)) & Int64(0xFFFFF))
                            dst = cute.make_ptr(self.activation_dtype,
                                destination_data.toint() + Int64(destination_row) * Int64(self.bytes_per_token),
                                AddressSpace.gmem, assumed_align=16)
                            with cute.arch.elect_one():
                                cp_async_bulk_s2g(dst, stage.iterator, Int32(activation_bytes))
                            cute.arch.sync_warp()
                            sf_row_view = destination_sf[Int64(sf_row), ((None, None), None)]
                            sf_values = cute.group_modes(cute.slice_(sf_row_view, (0, None, None)), 0, 2)
                            sf_vectors = cute.zipped_divide(sf_values, (4,))
                            for sf_round in cutlass.range_constexpr(ceil_div(cute.size(sf_vectors, mode=[1]), 32)):
                                vector = Int32(sf_round * 32) + lane_idx
                                if vector < Int32(cute.size(sf_vectors, mode=[1])):
                                    cute.copy(sf_atom, source_sf_vectors[None, vector], sf_vectors[None, vector])
                # Every batch has issued its stores before any ready is
                # published. All destinations are distinct pool rows. Their S2G
                # copies may share one bulk async-group without write-order
                # dependencies. Keep both source stages immutable until the
                # full wait below; .read alone is insufficient for ready.
                cute.arch.sync_warp()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0)
                # Join every bulk issuer and every generic SF writer before
                # the elected lane publishes any destination's release flag.
                cute.arch.sync_warp()
                for route_batch in cutlass.range(route_batch_count, unroll=1):
                    batch_record = record
                    if cutlass.const_expr(self.topk > 32):
                        if route_batch != Int32(0):
                            batch_record = self._load_input_dedup_route_batch(
                                routes, source_global_token, route_batch
                            )
                    for slot in cutlass.range_constexpr(min(self.topk, 32)):
                        lo = cute.arch.shuffle_sync(Int32(batch_record & Int64(0xFFFFFFFF)), Int32(slot))
                        hi = cute.arch.shuffle_sync(Int32(batch_record >> Int64(32)), Int32(slot))
                        route = (Int64(hi) << Int64(32)) | (Int64(lo) & Int64(0xFFFFFFFF))
                        if route != Int64(0):
                            ready_slot = Int32((route >> Int64(40)) & Int64(0xFFFFF))
                            flags = flags.accumulate(
                                Int32(0), self.token_in_flag_batch, (ready + ready_slot).toint()
                            )
                            cute.arch.sync_warp()
                pull_phase = pull_phase ^ Int32(1)
            work = work + global_warp_count
        flags.fire()
        cute.arch.sync_warp()
        iket.range_pop()

    @cute.jit
    def token_in(
        self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer,
    ) -> None:
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
        source_sizes = cute.make_tensor(sizes.iterator + owner_expert_begin, cute.make_layout((self.experts_per_rank,)))
        copy_elements = 4 if self.experts_per_rank % 4 == 0 else 1
        source_size_vectors = cute.zipped_divide(
            (_mark_alignment(source_sizes, 16) if cutlass.const_expr(copy_elements == 4) else source_sizes),
            (copy_elements,),
        )
        destination_size_vectors = cute.zipped_divide(owned_sizes, (copy_elements,))
        size_vector_count = cute.size(destination_size_vectors, mode=[1])
        size_copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL if copy_elements == 4 else cute.nvgpu.LoadCacheMode.ALWAYS
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
                    size_copy_atom, source_size_vectors[None, vector_idx], destination_size_vectors[None, vector_idx]
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
        if cutlass.const_expr(self.token_back_mode == "standalone_warps"):  # nosec B105
            sizes_ready_barrier = pipeline.NamedBarrier(
                barrier_id=self.standalone_size_barrier_id, num_threads=2 * self.transfer_thread_count
            )
            sizes_ready_barrier.arrive()

        if cutlass.const_expr(self.input_dedup_mode != "off"):
            self._token_in_deduplicated(smem_workspace, smem_base)
        if cutlass.const_expr(self.input_dedup_mode == "off"):
            iket.range_push("token_in.pull_payload")
            token_in_mbarriers = smem_workspace.ptr(self.token_in_mbarrier_region, smem_base)
            token_in_activation = smem_workspace.tensor(self.token_in_activation_smem_region, smem_base)
            warp_mbarrier = token_in_mbarriers + transfer_warp_idx
            warp_activation_stage = token_in_activation[transfer_warp_idx, None]
            if cutlass.const_expr(self.activation_is_quantized):
                token_in_sf = smem_workspace.tensor(self.token_in_sf_smem_region, smem_base)
                warp_sf_stage = token_in_sf[transfer_warp_idx, (None, None)]
            if lane_idx == Int32(0):
                cute.arch.mbarrier_init(warp_mbarrier, 1)
            cute.arch.sync_warp()

            if cutlass.const_expr(self.external_dispatch_pool):
                fc1_activation_pointer = self._token_comm_args.fc1_activation_pool.iterator
            else:
                fc1_activation_pointer = self._device_workspace.ptr(self.fc1_activation_region)
            fc1_ready_counter = self._device_workspace.ptr(self.fc1_ready_region)
            activation_bytes = cute.cosize(warp_activation_stage) * int(self.activation_dtype.width) // 8
            activation_sf_bytes = 0
            if cutlass.const_expr(self.activation_is_quantized):
                fc1_activation_sf = self.fc1_activation_sf_tensor(self._device_workspace)
                activation_sf_bytes = cute.cosize(warp_sf_stage) * int(self.activation_sf_dtype.width) // 8
                sf_copy_elements = 4
                source_sf_values = cute.slice_(warp_sf_stage, (0, None))
                source_sf_vectors = cute.zipped_divide(source_sf_values, (sf_copy_elements,))
                sf_copy_atom = _copy_atom(self.activation_sf_dtype, sf_copy_elements * int(self.activation_sf_dtype.width))

            next_dense_token = global_warp_idx
            expert_valid_begin = Int32(0)
            if cutlass.const_expr(self.activation_is_quantized):
                expert_sf_begin = Int32(0)
            expert_ready_slot_begin = Int32(0)
            pull_phase = Int32(0)
            flag_tracker = make_flag_batch_tracker(
                use_async=self.token_in_async_flags,
                flag_address=Int64(0),
                accumulated_flags=Int32(0),
                phase=Int32(0),
                thread_idx=lane_idx,
            )

            local_expert = Int32(0)
            while local_expert < Int32(self.experts_per_rank):
                expert_token_count = owned_sizes[local_expert]
                pool_expert_base = pool_expert_bases[local_expert]
                expert_valid_end = expert_valid_begin + expert_token_count
                pull_count = Int32(0)
                if next_dense_token < expert_valid_end:
                    pull_count = (expert_valid_end - next_dense_token + global_warp_count - Int32(1)) // global_warp_count

                for pull_round in cutlass.range(pull_count, unroll=1):
                    dense_token_idx = next_dense_token + Int32(pull_round) * global_warp_count
                    token_in_expert = dense_token_idx - expert_valid_begin
                    pool_token_idx = pool_expert_base + token_in_expert
                    ready_slot_idx = expert_ready_slot_begin + token_in_expert // Int32(self.tokens_per_fc1_ready_slot)

                    metadata = TokenSrcMetadata.load(
                        token_metadata_pointer.toint() + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                    )
                    peer_offset = self._token_comm_args.peer_rank_ptr_mapper.map(Int64(0), metadata.src_rank, Int64(0))
                    remote_activation_address = (
                        self._token_comm_args.activation.iterator.toint()
                        + peer_offset
                        + Int64(metadata.src_token) * Int64(self.bytes_per_token)
                    )
                    if cutlass.const_expr(self.activation_is_quantized):
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

                    if cutlass.const_expr(self.token_in_stage_split):
                        # Chunked pull: the halved stage bounces each token row in two pieces,
                        # trading one extra NVLink round trip per token for SMEM.
                        for chunk_idx in cutlass.range_constexpr(self._token_in_stage_chunk_count):
                            chunk_offset = Int64(chunk_idx) * Int64(activation_bytes)
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(warp_mbarrier, Int32(activation_bytes))
                                tma_load_1d(
                                    warp_activation_stage.iterator,
                                    Int64(remote_activation_address) + chunk_offset,
                                    warp_mbarrier,
                                    Int32(activation_bytes),
                                )
                            cute.arch.sync_warp()
                            cute.arch.mbarrier_wait(warp_mbarrier, pull_phase)
                            pull_phase = pull_phase ^ Int32(1)

                            chunk_destination = cute.make_ptr(
                                self.activation_dtype,
                                fc1_activation_pointer.toint()
                                + Int64(pool_token_idx) * Int64(self.bytes_per_token)
                                + chunk_offset,
                                AddressSpace.gmem,
                                assumed_align=16,
                            )
                            with cute.arch.elect_one():
                                cp_async_bulk_s2g(chunk_destination, warp_activation_stage.iterator, Int32(activation_bytes))
                            cute.arch.sync_warp()
                            cute.arch.cp_async_bulk_commit_group()
                            # Drain before the stage is reused by the next chunk (and before the flag).
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                            cute.arch.sync_warp()
                    else:
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
                            if cutlass.const_expr(self.activation_is_quantized):
                                tma_load_1d(
                                    warp_sf_stage.iterator, remote_sf_row.iterator, warp_mbarrier, Int32(activation_sf_bytes)
                                )
                        cute.arch.sync_warp()
                        cute.arch.mbarrier_wait(warp_mbarrier, pull_phase)

                        destination_activation = cute.make_ptr(
                            self.activation_dtype,
                            fc1_activation_pointer.toint() + Int64(pool_token_idx) * Int64(self.bytes_per_token),
                            AddressSpace.gmem,
                            assumed_align=16,
                        )
                        with cute.arch.elect_one():
                            cp_async_bulk_s2g(destination_activation, warp_activation_stage.iterator, Int32(activation_bytes))
                        cute.arch.sync_warp()
                        cute.arch.cp_async_bulk_commit_group()

                    if cutlass.const_expr(self.activation_is_quantized):
                        sf_token_idx = expert_sf_begin + token_in_expert
                        destination_sf_row = fc1_activation_sf[Int64(sf_token_idx), ((None, None), None)]
                        destination_sf_values = cute.slice_(destination_sf_row, (0, None, None))
                        destination_sf_values = cute.group_modes(destination_sf_values, 0, 2)
                        destination_sf_vectors = cute.zipped_divide(destination_sf_values, (sf_copy_elements,))
                        sf_vector_count = cute.size(destination_sf_vectors, mode=[1])
                        for sf_round in cutlass.range_constexpr(ceil_div(sf_vector_count, 32)):
                            sf_vector_idx = Int32(sf_round * 32) + lane_idx
                            if sf_vector_idx < Int32(sf_vector_count):
                                cute.copy(
                                    sf_copy_atom,
                                    source_sf_vectors[None, sf_vector_idx],
                                    destination_sf_vectors[None, sf_vector_idx],
                                )

                    if cutlass.const_expr(self.token_in_flag_batch == 1):
                        cute.arch.cp_async_bulk_wait_group(0)
                    elif cutlass.const_expr(not self.token_in_stage_split):
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    cute.arch.sync_warp()
                    ready_address = (fc1_ready_counter + ready_slot_idx).toint()
                    flag_tracker = flag_tracker.accumulate(Int32(0), self.token_in_flag_batch, ready_address)
                    cute.arch.sync_warp()
                    if cutlass.const_expr(not self.token_in_stage_split):
                        pull_phase = pull_phase ^ Int32(1)

                next_dense_token = next_dense_token + pull_count * global_warp_count
                expert_valid_begin = expert_valid_end
                if cutlass.const_expr(self.activation_is_quantized):
                    expert_sf_begin = expert_sf_begin + (
                        (expert_token_count + Int32(self.sf_padding_block - 1)) // Int32(self.sf_padding_block)
                    ) * Int32(self.sf_padding_block)
                expert_ready_slot_begin = expert_ready_slot_begin + (
                    (expert_token_count + Int32(self.tokens_per_fc1_ready_slot - 1))
                    // Int32(self.tokens_per_fc1_ready_slot)
                )
                local_expert = local_expert + Int32(1)

            flag_tracker.fire()
            cute.arch.sync_warp()
            iket.range_pop()
        if cutlass.const_expr(self.token_back_enabled and self.token_back_mode != "standalone_warps"):  # nosec B105
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
            sleep_cycles = cutlass.min(current_window - reference_window, Int32(self.standalone_max_backoff_cycles))
        if sleep_cycles > Int32(0):
            nanosleep(sleep_cycles)

    @cute.jit
    def _adaptive_pace(self, average_window: Int32, current_window: Int32, low_window: int, high_window: int) -> Int32:
        sleep_cycles = Int32(0)
        if current_window > average_window:
            average_window = average_window + ((current_window - average_window + Int32(3)) // Int32(4))
            sleep_cycles = current_window - average_window
            if sleep_cycles > Int32(high_window):
                sleep_cycles = Int32(high_window)
        else:
            average_window = average_window - ((average_window - current_window + Int32(3)) // Int32(4))
            sleep_cycles = average_window - current_window
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
            self.combine_format.is_quantized and self._token_comm_args.pre_reduced_activation_sf is None
        ):
            raise ValueError("Quantized token-back requires a scale destination.")

        global_worker_idx = self._linear_cta_idx * Int32(self.transfer_warp_count) + transfer_warp_idx
        global_worker_count = Int32(self.promised_launchable_sm_count * self.transfer_warp_count)
        token_back_mbarriers = smem_workspace.ptr(self.token_back_mbarrier_region, smem_base)
        worker_mbarrier = token_back_mbarriers + transfer_warp_idx
        if cutlass.const_expr(self.token_back_push_data):
            token_back_activation = smem_workspace.tensor(self.token_back_activation_smem_region, smem_base)
            worker_activation_stage = token_back_activation[transfer_warp_idx, None]
            activation_chunk_bytes = (
                cute.cosize(worker_activation_stage) * int(self.combine_format.act_dtype.width) // 8
            )
        if cutlass.const_expr(self.token_back_push_sf):
            token_back_sf = smem_workspace.tensor(self.token_back_sf_smem_region, smem_base)
            worker_sf_stage = token_back_sf[transfer_warp_idx, (None, None)]
            sf_chunk_bytes = cute.cosize(worker_sf_stage) * int(self.combine_format.scale_dtype.width) // 8
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(worker_mbarrier, 1)
        cute.arch.sync_warp()

        owned_sizes = smem_workspace.tensor(self.expert_sizes_smem_region, smem_base)
        if cutlass.const_expr(self.token_back_mode == "standalone_warps"):  # nosec B105
            sizes_ready_barrier = pipeline.NamedBarrier(
                barrier_id=self.standalone_size_barrier_id, num_threads=2 * self.transfer_thread_count
            )
            sizes_ready_barrier.arrive_and_wait()

        pool_expert_bases = self._router.pool_expert_base_tensor(self._device_workspace)
        token_metadata_pointer = self._router.token_src_metadata_pointer(self._device_workspace)
        fc2_done = self._device_workspace.ptr(self.fc2_done_region)
        if cutlass.const_expr(self.token_back_push_data):
            fc2_activation_pointer = self._device_workspace.ptr(self.fc2_activation_region)
            output_token_bytes = self.hidden_size * int(self.combine_format.act_dtype.width) // 8
            activation_chunk_count = ceil_div(output_token_bytes, activation_chunk_bytes)
            data_window_unit = ceil_div(activation_chunk_bytes * 2, 3)
            reuse_data_pacing_enabled = (
                self.token_back_mode == "reuse_dispatch_warps" and data_window_unit > self.minimum_pacing_window_cycles  # nosec B105
            )
            # Preserve the empirical low:initial:high ratio of 1:2.5:5.
            data_average_window = Int32(data_window_unit)
            data_low_window = data_window_unit * 2 // 5
            data_high_window = data_window_unit * 2
            print(f"[{data_window_unit}, {data_low_window}, {data_high_window}]")
        if cutlass.const_expr(self.token_back_push_sf):
            fc2_sf_pointer = self._device_workspace.ptr(self.fc2_activation_sf_region)
            output_sf_bytes = self.combine_sf_hidden_padded * int(self.combine_format.scale_dtype.width) // 8
            sf_chunk_count = ceil_div(output_sf_bytes, sf_chunk_bytes)
            sf_window_unit = ceil_div(sf_chunk_bytes * 2, 3)
            reuse_sf_pacing_enabled = (
                self.token_back_mode == "reuse_dispatch_warps" and sf_window_unit > self.minimum_pacing_window_cycles  # nosec B105
            )
            sf_average_window = Int32(sf_window_unit)
            sf_low_window = sf_window_unit * 2 // 5
            sf_high_window = sf_window_unit * 2

        next_dense_token = global_worker_idx - global_worker_count
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):  # nosec B105
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
                token_tile_count = (expert_token_count + Int32(self.tokens_per_fc1_ready_slot - 1)) // Int32(
                    self.tokens_per_fc1_ready_slot
                )
                completion_target = token_tile_count * Int32(self.fc2_done_signals_per_token_tile)
                iket.range_push("token_back.wait_fc2")
                while cute.arch.load(fc2_done + local_expert, Int32, sem="acquire", scope="gpu") < completion_target:
                    nanosleep(500)
                iket.range_pop()

            while next_dense_token < expert_valid_end:
                token_in_expert = next_dense_token - expert_valid_begin
                pool_token_idx = pool_expert_bases[local_expert] + token_in_expert
                metadata = TokenSrcMetadata.load(
                    token_metadata_pointer.toint() + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                )
                destination_topk = metadata.src_topk
                if cutlass.const_expr(self.reduce_topk_in_kernel):
                    destination_topk = Int32(0)
                peer_offset = self._token_comm_args.peer_rank_ptr_mapper.map(Int64(0), metadata.src_rank, Int64(0))
                is_remote_token = metadata.src_rank != self._local_rank

                if cutlass.const_expr(self.token_back_push_data):
                    iket.range_push("token_back.push_data")
                    local_activation_address = fc2_activation_pointer.toint() + Int64(pool_token_idx) * Int64(
                        output_token_bytes
                    )
                    remote_activation = cute.make_tensor(
                        cute.make_ptr(
                            self._token_comm_args.pre_reduced_activation.dtype,
                            self._token_comm_args.pre_reduced_activation.iterator.toint() + peer_offset,
                            AddressSpace.gmem,
                            assumed_align=(self._token_comm_args.pre_reduced_activation.iterator.max_alignment),
                        ),
                        self._token_comm_args.pre_reduced_activation.layout,
                    )
                    destination_row = remote_activation[Int64(metadata.src_token), destination_topk, None]
                    for chunk_idx in cutlass.range_constexpr(activation_chunk_count):
                        chunk_byte_offset = Int64(chunk_idx * activation_chunk_bytes)
                        chunk_bytes_this_round = min(
                            activation_chunk_bytes, output_token_bytes - chunk_idx * activation_chunk_bytes
                        )
                        current_chunk_bytes = Int32(chunk_bytes_this_round)
                        current_window_unit = ceil_div(chunk_bytes_this_round * 2, 3)
                        stateless_data_pacing_enabled = (
                            self.token_back_mode != "reuse_dispatch_warps"  # nosec B105
                            and current_window_unit > self.minimum_pacing_window_cycles
                        )
                        round_start_clock = Int64(0)
                        if cutlass.const_expr(reuse_data_pacing_enabled or stateless_data_pacing_enabled):
                            if is_remote_token:
                                round_start_clock = read_clock64()
                            else:
                                round_start_clock = round_start_clock
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(worker_mbarrier, current_chunk_bytes)
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
                                    destination_chunk, worker_activation_stage.iterator, current_chunk_bytes
                                )
                            else:
                                cp_async_bulk_s2g(
                                    destination_chunk, worker_activation_stage.iterator, current_chunk_bytes
                                )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        transfer_phase = transfer_phase ^ Int32(1)
                        if cutlass.const_expr(reuse_data_pacing_enabled):
                            if is_remote_token:
                                current_window = Int32(read_clock64() - round_start_clock)
                                data_average_window = self._adaptive_pace(
                                    data_average_window, current_window, data_low_window, data_high_window
                                )
                        elif cutlass.const_expr(stateless_data_pacing_enabled):
                            if is_remote_token:
                                current_window = Int32(read_clock64() - round_start_clock)
                                self._stateless_pace(Int32(current_window_unit), current_window)
                    iket.range_pop()

                if cutlass.const_expr(self.token_back_push_sf):
                    iket.range_push("token_back.push_sf")
                    local_sf_address = fc2_sf_pointer.toint() + Int64(pool_token_idx) * Int64(output_sf_bytes)
                    remote_sf = cute.make_tensor(
                        cute.make_ptr(
                            self._token_comm_args.pre_reduced_activation_sf.dtype,
                            self._token_comm_args.pre_reduced_activation_sf.iterator.toint() + peer_offset,
                            AddressSpace.gmem,
                            assumed_align=(self._token_comm_args.pre_reduced_activation_sf.iterator.max_alignment),
                        ),
                        self._token_comm_args.pre_reduced_activation_sf.layout,
                    )
                    destination_sf_row = remote_sf[Int64(metadata.src_token), destination_topk, None]
                    for chunk_idx in cutlass.range_constexpr(sf_chunk_count):
                        chunk_byte_offset = Int64(chunk_idx * sf_chunk_bytes)
                        chunk_bytes_this_round = min(sf_chunk_bytes, output_sf_bytes - chunk_idx * sf_chunk_bytes)
                        current_chunk_bytes = Int32(chunk_bytes_this_round)
                        current_window_unit = ceil_div(chunk_bytes_this_round * 2, 3)
                        stateless_sf_pacing_enabled = (
                            self.token_back_mode != "reuse_dispatch_warps"  # nosec B105
                            and current_window_unit > self.minimum_pacing_window_cycles
                        )
                        round_start_clock = Int64(0)
                        if cutlass.const_expr(reuse_sf_pacing_enabled or stateless_sf_pacing_enabled):
                            if is_remote_token:
                                round_start_clock = read_clock64()
                            else:
                                round_start_clock = round_start_clock
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(worker_mbarrier, current_chunk_bytes)
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
                            cp_async_bulk_s2g(destination_chunk, worker_sf_stage.iterator, current_chunk_bytes)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        transfer_phase = transfer_phase ^ Int32(1)
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

    @cute.jit
    def next_token(self, current_token: Int32) -> Int32:
        global_worker_count = self.promised_launchable_sm_count * self.transfer_warp_count
        schedule_counter = None
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):  # nosec B105
            schedule_counter = self._device_workspace.ptr(self.token_back_schedule_region)
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):  # nosec B105
            claimed_token = Int32(0)
            if self._lane_idx == Int32(0):
                claimed_token = cute.arch.atomic_add(schedule_counter, Int32(1), sem="relaxed", scope="gpu")
            return Int32(cute.arch.shuffle_sync(claimed_token, Int32(0)))
        return current_token + global_worker_count

    @cute.jit
    def reset_tail(self) -> None:
        """Reset communication state with all configured token-in transfer warps."""
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
        global_reset_thread = self._linear_cta_idx * Int32(self.transfer_thread_count) + transfer_thread_idx
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
            prologue_grid_sync=False,
            epilogue_grid_sync=True,
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


__all__ = ["TokenCommDeterministic"]
