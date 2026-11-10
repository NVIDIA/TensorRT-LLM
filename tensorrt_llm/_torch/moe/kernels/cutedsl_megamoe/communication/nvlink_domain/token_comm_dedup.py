"""Combine-deduplicating token communication: dispatch as in token_comm, combine reduces a token's expert copies
on the destination rank and pushes one row per (src_rank, src_token)."""

import dataclasses
import math
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
from ...helpers.software_sync import NvlinkBarrier
from ...helpers.ptx_helpers import (
    cp_async_bulk_s2g,
    cp_reduce_async_bulk_add_bf16_s2g,
    cp_reduce_async_bulk_add_u32_s2g,
    nanosleep,
    red_add_relaxed_sys_s32,
    cvt_rn_bf16x2_f32x2,
    f32x2_pack,
    fma_rn_f32x2_bf16x2,
    stg_b64,
    stg_f32,
    tma_load_1d,
)
from ...helpers.smem_workspace import SmemWorkspace
from ...helpers.utils import ceil_div, round_up
from ...quant_def import CombineFormat, QuantKind
from ..token_protocol import TokenSrcMetadata
from .symmetric_buffer import SymmetricBufferDevice


from .token_comm import TokenBackMode, TokenBackScheduleMode, TokenCommArgs  # noqa: E402



@dataclasses.dataclass(frozen=True)
class _CopyInfo:
    """Per-route order and count-minus-one, packed into two eight-bit fields."""

    dst_order_id: Int32
    dst_token_cnt: Int32

    FIELD_BITS: ClassVar[int] = 8
    FIELD_MASK: ClassVar[int] = (1 << FIELD_BITS) - 1

    def pack(self) -> Int32:
        return self.dst_order_id | ((self.dst_token_cnt - Int32(1)) << Int32(self.FIELD_BITS))

    @classmethod
    def unpack(cls, packed: Int32) -> "_CopyInfo":
        return cls(
            dst_order_id=packed & Int32(cls.FIELD_MASK),
            dst_token_cnt=((packed >> Int32(cls.FIELD_BITS)) & Int32(cls.FIELD_MASK)) + Int32(1),
        )


@dataclasses.dataclass(frozen=True)
class _SortedElement:
    """Sort-buffer element: Int128 ``[flat_topk_index | copy_info | topk_score | pad]`` with scores, else Int64
    ``[flat_topk_index | copy_info]``; ``flat_topk_index = token * topk + slot``."""

    flat_topk_index: Int32
    copy_info: Int32
    topk_score: Optional[cutlass.Float32]

    @staticmethod
    def storage_type(with_score: bool):
        return cutlass.Int128 if with_score else cutlass.Int64

    def pack(self) -> Union[cutlass.Int128, Int64]:
        if cutlass.const_expr(self.topk_score is None):
            scratch = cute.make_rmem_tensor((2,), cutlass.Int32)
            scratch[0] = self.flat_topk_index
            scratch[1] = self.copy_info
            return cute.recast_tensor(scratch, cutlass.Int64)[0]
        scratch = cute.make_rmem_tensor((4,), cutlass.Int32)
        scratch[0] = self.flat_topk_index
        scratch[1] = self.copy_info
        cute.recast_tensor(scratch, cutlass.Float32)[2] = self.topk_score
        return cute.recast_tensor(scratch, cutlass.Int128)[0]

    @classmethod
    def from_packed(cls, packed: Union[cutlass.Int128, Int64]) -> "_SortedElement":
        if cutlass.const_expr(type(packed).width == 64):
            scratch = cute.make_rmem_tensor((2,), cutlass.Int32)
            cute.recast_tensor(scratch, cutlass.Int64)[0] = packed
            return cls(flat_topk_index=scratch[0], copy_info=scratch[1], topk_score=None)
        scratch = cute.make_rmem_tensor((4,), cutlass.Int32)
        cute.recast_tensor(scratch, cutlass.Int128)[0] = packed
        return cls(
            flat_topk_index=scratch[0],
            copy_info=scratch[1],
            topk_score=cute.recast_tensor(scratch, cutlass.Float32)[2],
        )


def _odd_even_merge_sort_pairs(length: int) -> Tuple[Tuple[int, int], ...]:
    """Batcher odd-even merge sort as (low, high) compare-exchange pairs; generated for the next power of two and
    pruned to pairs inside ``length`` (a pruned partner would be +inf, a no-op)."""
    width = 1 << (max(length, 1) - 1).bit_length()
    pairs = []

    def merge(lo, n, r):
        step = r * 2
        if step < n:
            merge(lo, n, step)
            merge(lo + r, n, step)
            pairs.extend((i, i + r) for i in range(lo + r, lo + n - r, step))
        else:
            pairs.append((lo, lo + r))

    def sort(lo, n):
        if n > 1:
            half = n // 2
            sort(lo, half)
            sort(lo + half, half)
            merge(lo, n, 1)

    sort(0, width)
    return tuple((i, j) for i, j in pairs if j < length)


@cute.jit
def _copy_atom(dtype, num_bits_per_copy: int):
    return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=num_bits_per_copy)


class _DedupMetadataPushRouter(KernelComponent):
    """Sort and push routing metadata into each destination rank's final pool."""

    router_smem_limit_bytes: ClassVar[int] = 227 * 1024
    router_warps_per_cta: ClassVar[int] = 16
    # Supported top-k: one token's routes are sorted in registers by a per-thread network (191 compare-exchanges at 32).
    router_max_topk: ClassVar[int] = 32
    # Two Int32 registers per route (expert; within-expert index | copy info); the largest footprint without spills.
    router_max_elements_per_thread: ClassVar[int] = 24
    # SMEM reserved for the expert tables / mbarriers next to the per-route buffers.
    router_table_smem_bytes: ClassVar[int] = 24 * 1024

    sizes_by_rank_region = "nvlink.token_comm_dedup.sizes_by_rank"
    sizes_region = "nvlink.token_comm_dedup.sizes"
    sizes_ready_region = "nvlink.token_comm_dedup.sizes_ready"
    metadata_ready_region = "nvlink.token_comm_dedup.metadata_ready"
    sorted_metadata_region = "nvlink.token_comm_dedup.sorted_metadata"
    sorted_scores_region = "nvlink.token_comm_dedup.sorted_scores"
    pool_expert_base_region = "nvlink.token_comm_dedup.pool_expert_base"
    token_src_metadata_region = "nvlink.token_comm_dedup.token_src_metadata"  # nosec B105
    fc1_topk_scores_region = "nvlink.token_comm_dedup.fc1_topk_scores"
    source_expert_base_region = "nvlink.token_comm_dedup.source_expert_base"
    push_destination_base_region = "nvlink.token_comm_dedup.push_destination_base"
    sorted_metadata_ready_region = "nvlink.token_comm_dedup.sorted_metadata_ready"
    push_table_ready_region = "nvlink.token_comm_dedup.push_table_ready"
    router_size_counter_region = "nvlink.token_comm_dedup.router_size_counter"
    router_histogram_done_region = "nvlink.token_comm_dedup.router_histogram_done"
    source_base_ready_region = "nvlink.token_comm_dedup.source_base_ready"
    valid_rank_mask_region = "nvlink.token_comm_dedup.valid_rank_mask"

    router_data_histogram_region = "nvlink.token_comm_dedup.router_smem.data_histogram"
    router_data_prefix_region = "nvlink.token_comm_dedup.router_smem.data_prefix"
    router_data_warp_totals_region = "nvlink.token_comm_dedup.router_smem.data_warp_totals"
    router_data_sorted_region = "nvlink.token_comm_dedup.router_smem.data_sorted"
    router_data_scores_region = "nvlink.token_comm_dedup.router_smem.data_scores"
    router_data_base_region = "nvlink.token_comm_dedup.router_smem.data_base"
    router_helper_size_matrix_region = "nvlink.token_comm_dedup.router_smem.helper_size_matrix"
    router_helper_totals_region = "nvlink.token_comm_dedup.router_smem.helper_totals"
    router_helper_prefix_region = "nvlink.token_comm_dedup.router_smem.helper_prefix"
    router_push_expert_base_region = "nvlink.token_comm_dedup.router_smem.push_expert_base"
    router_push_destination_base_region = "nvlink.token_comm_dedup.router_smem.push_destination_base"
    router_push_warp_start_counts_region = "nvlink.token_comm_dedup.router_smem.push_warp_start_counts"
    router_helper_warp_totals_region = "nvlink.token_comm_dedup.router_smem.helper_warp_totals"
    router_helper_load_mbarrier_region = "nvlink.token_comm_dedup.router_smem.helper_load_mbarrier"
    router_data_load_mbarrier_region = "nvlink.token_comm_dedup.router_smem.data_load_mbarrier"

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
        self.carry_topk_scores = True

        self.token_padding_block = impl_desc["token_padding_block"]
        self.promised_launchable_sm_count = impl_desc["promised_launchable_sm_count"]
        self.router_smem_limit_bytes = impl_desc.get("router_smem_limit_bytes", 227 * 1024)

        self._validate_router_configuration()
        # Large topk: drop warps so ``threads * topk`` routes still fit the per-CTA sort capacity.
        while self.router_warps_per_cta > 1 and self.router_warps_per_cta * 32 * self.topk > self.router_max_routes_per_cta:
            self.router_warps_per_cta //= 2
        if 32 * self.topk > self.router_max_routes_per_cta:
            raise ValueError(
                f"router_smem_limit_bytes={self.router_smem_limit_bytes} leaves room for {self.router_max_routes_per_cta} "
                f"routes per CTA, below the 32 x topk={32 * self.topk} that one warp holding one token per thread needs."
            )
        self.expert_count_padded = round_up(self.expert_count, 4)
        self.expert_count_with_trash = self.expert_count_padded + 1
        self.router_elements_per_lane, self.router_data_cta_count = self._router_launch_configuration()
        # Historical name: this field counts routes (tokens_per_cta * topk) rather than source tokens.
        self.router_tokens_per_cta = self.router_elements_per_lane * self.router_warps_per_cta * 32
        # Same grid footprint as the non-dedup router (DATA by route capacity, PUSH by experts / warps).
        self.router_push_cta_count = min(ceil_div(self.expert_count, self.router_warps_per_cta),
                                         self.promised_launchable_sm_count)
        self.router_grid_cta_count = max(self.router_data_cta_count + 1, self.router_push_cta_count)
        if self.router_grid_cta_count > self.promised_launchable_sm_count:
            raise ValueError(
                "Router grid exceeds promised_launchable_sm_count; all metadata-push CTAs must be concurrently resident."
            )
        self.worst_case_token_count = self.worst_case_padded_tokens(self.token_padding_block)
        self._router_smem_workspace = self._build_router_smem_workspace()

        self._device_workspace = None
        self._peer_rank_ptr_mapper = None
        self._router_local_rank = None
        self._router_thread_idx = None
        self._router_linear_cta_idx = None
        self._router_warp_idx = None
        self._router_lane_idx = None
        self._router_smem_base = None

    def _validate_router_configuration(self) -> None:
        if self.topk > self.router_max_topk:
            raise ValueError(f"Combine dedup sorts a token's routes in registers; topk must be <= {self.router_max_topk}.")
        if self.world_size > 32:
            raise ValueError("Combine dedup keeps the destination-rank mask in an Int32; world_size must be <= 32.")
        if self.max_tokens_per_rank > TokenSrcMetadata.MAX_TOKENS:
            raise ValueError(
                f"TokenSrcMetadata keeps the token index in {TokenSrcMetadata.TOKEN_BITS} bits; "
                f"max_tokens_per_rank must be <= {TokenSrcMetadata.MAX_TOKENS}."
            )
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
            raise ValueError(f"topk must not exceed expert_count, got {self.topk} and {self.expert_count}.")

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.world_size

    def worst_case_padded_tokens(self, block: int) -> int:
        source_token_capacity = self.world_size * self.max_tokens_per_rank
        routes_per_source_token = min(self.topk, self.experts_per_rank)
        route_capacity = source_token_capacity * routes_per_source_token
        active_expert_capacity = min(self.experts_per_rank, route_capacity)
        route_budget_blocks = active_expert_capacity + (route_capacity - active_expert_capacity) // block
        expert_bound_blocks = active_expert_capacity * int(ceil_div(source_token_capacity, block))
        return min(route_budget_blocks, expert_bound_blocks) * block

    @property
    def router_max_routes_per_cta(self) -> int:
        # Per-route SMEM: sort record (16 B with scores, 8 B without) + fp32 score slab; capped like the base router.
        record_bytes = int(_SortedElement.storage_type(self.carry_topk_scores).width) // 8
        per_route = max(record_bytes, 8) + (4 if self.carry_topk_scores else 0)
        return min(16384, (self.router_smem_limit_bytes - self.router_table_smem_bytes) // per_route)

    def _router_launch_configuration(self) -> Tuple[int, int]:
        def next_power_of_two(value: int) -> int:
            return 1 << (max(value, 1) - 1).bit_length()

        # DATA grid = routes / capacity; each thread owns whole tokens so a token's copies never straddle threads.
        routed_token_capacity = self.max_tokens_per_rank * self.topk
        threads_per_cta = self.router_warps_per_cta * 32
        minimum_cta_capacity = 2048
        cta_capacity = min(
            self.router_max_routes_per_cta, next_power_of_two(max(routed_token_capacity, minimum_cta_capacity))
        )
        # Whole tokens per thread, bounded by SMEM capacity and register footprint; at least one token.
        tokens_per_thread = max(
            1, min(cta_capacity // (threads_per_cta * self.topk), self.router_max_elements_per_thread // self.topk)
        )
        self.router_tokens_per_thread = tokens_per_thread
        if tokens_per_thread * threads_per_cta * self.topk > 1 << 16:
            raise ValueError("The within-expert index is packed into 16 bits; routes per DATA CTA must be <= 65536.")
        elements_per_lane = tokens_per_thread * self.topk
        data_cta_count = ceil_div(self.max_tokens_per_rank, tokens_per_thread * threads_per_cta)
        # Slab-read bank swizzle (ROUTER_SMEM_BANK_CONFLICT_PLAN.md section 4): lane l reads vector chunk
        # ``c XOR f(l)``, f(l) = (l // (32 / gcd(K, 32))) % (gcd(K, 32) / V), which makes every wavefront cover the
        # 32 banks once; ``_router_value_coordinate`` inverts it.
        k = self.topk
        self.router_load_vector_words = 4 if k % 4 == 0 else (2 if k % 2 == 0 else 1)
        gcd_k = math.gcd(k, 32)
        self.router_swizzle_period_lanes = 32 // gcd_k
        self.router_swizzle_groups = max(1, gcd_k // self.router_load_vector_words)
        assert (k // self.router_load_vector_words) % self.router_swizzle_groups == 0
        return elements_per_lane, data_cta_count

    @cute.jit
    def _router_slot_swizzle(self) -> Int32:
        """f(lane) of the slab-read swizzle: the XOR applied to a thread's vector-chunk index (0 when identity)."""
        if cutlass.const_expr(self.router_swizzle_groups == 1):
            return Int32(0)
        lane = self._router_thread_idx % Int32(32)
        return (lane // Int32(self.router_swizzle_period_lanes)) % Int32(self.router_swizzle_groups)

    def _build_router_smem_workspace(self) -> SmemWorkspace:
        workspace = SmemWorkspace()
        workspace.register_mbarrier(self.router_helper_load_mbarrier_region, 1)
        workspace.register_mbarrier(self.router_data_load_mbarrier_region, 1)
        overlay = workspace.create_overlay("nvlink.token_comm_dedup.router_smem.role")
        data_lifetime = overlay.add_lifetime("data_cta")
        data_lifetime.register_tensor(self.router_data_histogram_region, cutlass.Int32, (self.expert_count_with_trash,))
        data_lifetime.register_tensor(
            self.router_data_prefix_region, cutlass.Int32, (self.expert_count_with_trash,), byte_alignment=16
        )
        data_lifetime.register_tensor(self.router_data_warp_totals_region, cutlass.Int32, (self.router_warps_per_cta,))
        # The sort buffer doubles as the landing slab of the bulk-copied topk indices; scores get their own slab.
        sorted_element_bytes = int(_SortedElement.storage_type(self.carry_topk_scores).width) // 8
        slab_bytes = self.router_tokens_per_cta * 8
        sorted_count = max(self.router_tokens_per_cta, ceil_div(slab_bytes, sorted_element_bytes))
        data_lifetime.register_tensor(
            self.router_data_sorted_region,
            _SortedElement.storage_type(self.carry_topk_scores),
            (sorted_count,),
            byte_alignment=16,
        )
        if self.carry_topk_scores:
            data_lifetime.register_tensor(
                self.router_data_scores_region, cutlass.Float32, (self.router_tokens_per_cta,), byte_alignment=16
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
        # Dense PUSH tables: per-expert source / destination start positions plus one ballot slot per warp.
        push_lifetime = overlay.add_lifetime("push_cta")
        push_lifetime.register_tensor(
            self.router_push_expert_base_region, cutlass.Int32, (self.expert_count_padded,), byte_alignment=16
        )
        push_lifetime.register_tensor(
            self.router_push_destination_base_region, cutlass.Int32, (self.expert_count_padded,), byte_alignment=16
        )
        push_lifetime.register_tensor(
            self.router_push_warp_start_counts_region, cutlass.Int32, (self.router_warps_per_cta,), byte_alignment=16
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
            self.sizes_region, cutlass.Int32, (self.expert_count_padded,), buffer_space="shared", reset="tail_reset"
        )
        workspace.register(self.sizes_ready_region, cutlass.Int32, (1,), buffer_space="shared", reset="tail_reset")
        workspace.register(self.metadata_ready_region, cutlass.Int32, (1,), buffer_space="shared", reset="tail_reset")
        workspace.register(self.sorted_metadata_region, cutlass.Int64, (maximum_routed_tokens,), buffer_space="local")
        if self.carry_topk_scores:
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
        if self.carry_topk_scores:
            workspace.register(
                self.fc1_topk_scores_region, cutlass.Float32, (self.worst_case_token_count,), buffer_space="shared"
            )
        workspace.register(self.pool_expert_base_region, cutlass.Int32, (self.experts_per_rank,), buffer_space="local")
        # Per source token: bitmask of destination ranks holding >= 1 copy (rewritten every launch).
        workspace.register(self.valid_rank_mask_region, cutlass.Int32, (self.max_tokens_per_rank,), buffer_space="local")
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
                self.router_size_counter_region,
                cutlass.Int32,
                (self.expert_count_with_trash,),
                buffer_space="local",
                reset="tail_reset",
            )
            workspace.register(
                self.router_histogram_done_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
            )
            workspace.register(
                self.source_base_ready_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
            )

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "_DedupMetadataPushRouter":
        if values:
            raise ValueError("_DedupMetadataPushRouter carries no MLIR values.")
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
        if cutlass.const_expr(self.carry_topk_scores and topk_scores is None):
            raise ValueError("Combine dedup routing requires router topk_scores.")
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
        self._router_warp_idx = warp_idx
        self._router_lane_idx = lane_idx
        self._router_smem_base = smem_base

        if cutlass.const_expr(self.router_data_cta_count == 1):
            self._router_single_cta(topk_indices, topk_scores, smem_base)
        else:
            self._router_multiple_ctas(topk_indices, topk_scores, smem_base)
        if linear_cta_idx < Int32(self.router_push_cta_count):
            self._router_push_metadata(smem_base)

        device_workspace.remove_device_members()
        self._device_workspace = None
        self._peer_rank_ptr_mapper = None
        self._router_local_rank = None
        self._router_thread_idx = None
        self._router_linear_cta_idx = None
        self._router_warp_idx = None
        self._router_lane_idx = None
        self._router_smem_base = None

    @cute.jit
    def _router_single_cta(
        self, topk_indices: cute.Tensor, topk_scores: Optional[cute.Tensor], smem_base: cute.Pointer
    ) -> None:
        if self._router_linear_cta_idx < Int32(self.router_data_cta_count):
            block_thread_count = self.router_warps_per_cta * 32
            trash_bucket = self.expert_count_padded
            histogram = self._router_smem_workspace.tensor(self.router_data_histogram_region, smem_base)
            prefix = self._router_smem_workspace.tensor(self.router_data_prefix_region, smem_base)
            warp_totals = self._router_smem_workspace.tensor(self.router_data_warp_totals_region, smem_base)
            sorted_elements = self._router_smem_workspace.tensor(self.router_data_sorted_region, smem_base)

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
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
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

            # Copy fields are computed while the helper exchanges sizes and builds push tables.
            iket.range_push("router.sort")
            self._sort_token_routes_and_scatter(
                expert_registers, within_expert_indices, score_registers, sorted_elements, prefix, topk_indices.dtype
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
            prefix = self._router_smem_workspace.tensor(self.router_data_prefix_region, smem_base)
            warp_totals = self._router_smem_workspace.tensor(self.router_data_warp_totals_region, smem_base)
            sorted_elements = self._router_smem_workspace.tensor(self.router_data_sorted_region, smem_base)
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
                        cute.arch.atomic_add(size_counter + expert, histogram[expert], sem="relaxed", scope="gpu")
                    )
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.router_histogram_done_region), Int32(1), sem="release", scope="gpu"
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
            # Copy fields are computed while the helper exchanges sizes and builds push tables.
            iket.range_push("router.sort")
            self._sort_token_routes_and_scatter(
                expert_registers, within_expert_indices, score_registers, sorted_elements, prefix, topk_indices.dtype
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
            for expert_round in cutlass.range_constexpr(expert_round_count):
                expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_padded):
                    dump_base[expert] = dump_base[expert] + source_expert_base[expert]
            cute.arch.sync_threads()
            self._dump_router_output_by_expert(histogram, prefix, dump_base, sorted_elements)
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
        cute.arch.fence_proxy("async.shared", space="cta")
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
    def _router_push_metadata(self, smem_base: cute.Pointer) -> None:
        """Dense PUSH of the sorted records: each CTA sweeps a contiguous interval block-stride and derives a
        record's expert from its position by a forward scan over the SMEM prefix table."""
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

        source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
        global_destination_base = self._device_workspace.tensor(self.push_destination_base_region)
        source_sizes = self._device_workspace.tensor(self.sizes_by_rank_region)
        last_expert = Int32(self.expert_count - 1)
        # Invalid routes sit in the trash bucket after the last expert and are never pushed.
        total_routes = source_expert_base[last_expert] + source_sizes[Int32(self._router_local_rank), last_expert]
        # CTA c owns records [c * routes_per_cta, (c + 1) * routes_per_cta), a multiple of the block (coalesced).
        push_threads = Int32(self.router_push_cta_count * block_thread_count)
        routes_per_cta = (total_routes + push_threads - Int32(1)) // push_threads * Int32(block_thread_count)
        cta_begin = self._router_linear_cta_idx * routes_per_cta
        cta_end = cutlass.min(cta_begin + routes_per_cta, total_routes)

        # Stage the per-expert tables in SMEM and ballot the expert of this CTA's first record.
        expert_base = self._router_smem_workspace.tensor(self.router_push_expert_base_region, smem_base)
        push_destination_base = self._router_smem_workspace.tensor(self.router_push_destination_base_region, smem_base)
        warp_start_counts = self._router_smem_workspace.tensor(self.router_push_warp_start_counts_region, smem_base)
        warp_start_count = Int32(0)
        for expert_round in cutlass.range_constexpr(ceil_div(self.expert_count_padded, block_thread_count)):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            starts_before = Int32(0)
            if expert < Int32(self.expert_count_padded):
                expert_start = source_expert_base[expert]
                expert_base[expert] = expert_start
                push_destination_base[expert] = global_destination_base[expert]
                if expert <= last_expert:
                    if expert_start <= cta_begin:
                        starts_before = Int32(1)
            warp_start_count = warp_start_count + Int32(
                cute.arch.popc(cute.arch.vote_ballot_sync(starts_before != Int32(0)))
            )
        if self._router_lane_idx == Int32(0):
            warp_start_counts[self._router_warp_idx] = warp_start_count
        cute.arch.sync_threads()
        start_count = Int32(0)
        for warp in cutlass.range_constexpr(self.router_warps_per_cta):
            start_count = start_count + warp_start_counts[warp]
        expert = cutlass.max(start_count - Int32(1), Int32(0))

        source_metadata = self._device_workspace.ptr(self.sorted_metadata_region)
        destination_metadata_base = self._device_workspace.ptr(self.token_src_metadata_region).toint()
        if cutlass.const_expr(self.carry_topk_scores):
            source_scores = self._device_workspace.ptr(self.sorted_scores_region)
            destination_scores_base = self._device_workspace.ptr(self.fc1_topk_scores_region).toint()
        source_position = cta_begin + self._router_thread_idx
        # Per-expert state lives in registers and is refreshed only when the cursor advances.
        next_expert_start = Int32(0x7FFFFFFF)
        if expert < last_expert:
            next_expert_start = expert_base[expert + Int32(1)]
        while (expert < last_expert) and (next_expert_start <= source_position):
            expert = expert + Int32(1)
            if expert < last_expert:
                next_expert_start = expert_base[expert + Int32(1)]
            else:
                next_expert_start = Int32(0x7FFFFFFF)
        destination_shift = push_destination_base[expert] - expert_base[expert]
        peer_offset = self._peer_rank_ptr_mapper.map(Int64(0), expert // Int32(self.experts_per_rank), Int64(0))
        while source_position < cta_end:
            metadata = cute.arch.load(source_metadata + source_position, cutlass.Int64)
            if next_expert_start <= source_position:
                # Advance the cursor to the last expert whose first position is <= source_position.
                while (expert < last_expert) and (next_expert_start <= source_position):
                    expert = expert + Int32(1)
                    if expert < last_expert:
                        next_expert_start = expert_base[expert + Int32(1)]
                    else:
                        next_expert_start = Int32(0x7FFFFFFF)
                destination_shift = push_destination_base[expert] - expert_base[expert]
                peer_offset = self._peer_rank_ptr_mapper.map(Int64(0), expert // Int32(self.experts_per_rank), Int64(0))
            destination_position = source_position + destination_shift
            stg_b64(destination_metadata_base + peer_offset + Int64(destination_position) * Int64(TokenSrcMetadata.nbytes), metadata)
            if cutlass.const_expr(self.carry_topk_scores):
                score = cute.arch.load(source_scores + source_position, cutlass.Float32)
                stg_f32(destination_scores_base + peer_offset + Int64(destination_position) * Int64(4), score)
            source_position = source_position + Int32(block_thread_count)

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
            while cute.arch.load(sizes_ready, Int32, sem="acquire", scope="sys") != Int32(self.world_size):
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
                local_ring_position = (
                    Int32(self._router_local_rank) - destination_rank + Int32(self.world_size)
                ) % Int32(self.world_size)
                source_ring_offset = Int32(0)
                for ring_position in cutlass.range_constexpr(self.world_size):
                    source_rank = (destination_rank + Int32(ring_position)) % Int32(self.world_size)
                    if Int32(ring_position) < local_ring_position:
                        source_ring_offset = source_ring_offset + size_matrix[source_rank, global_expert]
                push_destination_base[global_expert] = destination_pool_base + source_ring_offset
        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            cute.arch.atomic_add(
                self._device_workspace.ptr(self.push_table_ready_region), Int32(1), sem="release", scope="gpu"
            )
        iket.range_pop()

    @cute.jit
    def _load_router_inputs(
        self, topk_indices: cute.Tensor, topk_scores: Optional[cute.Tensor]
    ) -> Tuple[cute.Tensor, Optional[cute.Tensor]]:
        """Bulk-copy this CTA's token slab into SMEM; thread ``tid`` then owns tokens ``tid, tid + threads, ...``
        (``topk`` contiguous routes each) in registers."""
        topk = self.topk
        threads_per_cta = self.router_warps_per_cta * 32
        tokens_per_thread = self.router_tokens_per_thread
        tokens_per_cta = tokens_per_thread * threads_per_cta
        routes_per_cta = tokens_per_cta * topk
        elements_per_thread = tokens_per_thread * topk
        experts_per_rank = self.experts_per_rank
        sentinel_expert = Int32(self.expert_count_padded)
        actual_token_count = Int32(self.max_tokens_per_rank)
        index_bytes = int(topk_indices.dtype.width) // 8
        maximum_elements = self.max_tokens_per_rank * topk

        cta_token_begin = self._router_linear_cta_idx * Int32(tokens_per_cta)
        cta_route_begin = cta_token_begin * Int32(topk)
        valid_routes = Int32(maximum_elements) - cta_route_begin
        if valid_routes > Int32(routes_per_cta):
            valid_routes = Int32(routes_per_cta)
        if valid_routes < Int32(0):
            valid_routes = Int32(0)

        # Index slab aliases the sorted buffer; the score slab is its own region (it must survive the sort).
        smem_base_address = self._router_smem_workspace.ptr(self.router_data_sorted_region, self._router_smem_base).toint()
        slab_indices = cute.make_tensor(
            cute.make_ptr(topk_indices.dtype, smem_base_address, AddressSpace.smem, assumed_align=16),
            cute.make_layout((routes_per_cta,)),
        )
        slab_scores = None
        if cutlass.const_expr(self.carry_topk_scores):
            slab_scores = self._router_smem_workspace.tensor(self.router_data_scores_region, self._router_smem_base)
        load_mbarrier = self._router_smem_workspace.ptr(self.router_data_load_mbarrier_region, self._router_smem_base)
        topk_flat = cute.make_tensor(topk_indices.iterator, cute.make_layout((maximum_elements,)))
        scores_flat = None
        if cutlass.const_expr(self.carry_topk_scores):
            scores_flat = cute.make_tensor(topk_scores.iterator, cute.make_layout((maximum_elements,)))

        if self._router_thread_idx == Int32(0):
            cute.arch.mbarrier_init(load_mbarrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # Bulk part: the largest 16 B multiple of the valid bytes; tail: the < 16 B remainder, one element per thread.
        index_bytes_valid = valid_routes * Int32(index_bytes)
        index_bytes_bulk = index_bytes_valid & Int32(~15)
        score_bytes_valid = valid_routes * Int32(4)
        score_bytes_bulk = score_bytes_valid & Int32(~15)
        expected_bytes = index_bytes_bulk
        if cutlass.const_expr(self.carry_topk_scores):
            expected_bytes = expected_bytes + score_bytes_bulk
        if self._router_thread_idx == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(load_mbarrier, expected_bytes)
            if index_bytes_bulk > Int32(0):
                tma_load_1d(
                    slab_indices.iterator,
                    cute.make_ptr(
                        topk_indices.dtype,
                        topk_flat.iterator.toint() + Int64(cta_route_begin) * Int64(index_bytes),
                        AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    load_mbarrier,
                    index_bytes_bulk,
                )
            if cutlass.const_expr(self.carry_topk_scores):
                if score_bytes_bulk > Int32(0):
                    tma_load_1d(
                        slab_scores.iterator,
                        cute.make_ptr(
                            cutlass.Float32,
                            scores_flat.iterator.toint() + Int64(cta_route_begin) * Int64(4),
                            AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        load_mbarrier,
                        score_bytes_bulk,
                    )
        index_tail_begin = index_bytes_bulk // Int32(index_bytes)
        if self._router_thread_idx < valid_routes - index_tail_begin:
            route = index_tail_begin + self._router_thread_idx
            slab_indices[route] = topk_flat[cta_route_begin + route]
        if cutlass.const_expr(self.carry_topk_scores):
            score_tail_begin = score_bytes_bulk // Int32(4)
            if self._router_thread_idx < valid_routes - score_tail_begin:
                route = score_tail_begin + self._router_thread_idx
                slab_scores[route] = scores_flat[cta_route_begin + route]
        cute.arch.mbarrier_wait(load_mbarrier, 0)
        cute.arch.sync_threads()

        # Token t of this thread = routes [(t * threads + tid) * topk, +topk); only expert ids go to registers.
        expert_registers = cute.make_rmem_tensor((elements_per_thread,), cutlass.Int32)
        slab_indices_i32 = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, smem_base_address, AddressSpace.smem, assumed_align=16),
            cute.make_layout((routes_per_cta * (index_bytes // 4),)),
        )
        # Register s holds physical slot ``(chunk(s) XOR swizzle) * V + s % V``; _router_value_coordinate inverts it.
        vector_words: cutlass.Constexpr[int] = self.router_load_vector_words
        swizzle = self._router_slot_swizzle()
        for token_slot in cutlass.range_constexpr(tokens_per_thread):
            local_token = Int32(token_slot * threads_per_cta) + self._router_thread_idx
            token_idx = cta_token_begin + local_token
            token_valid = token_idx < actual_token_count
            slab_route_begin = local_token * Int32(topk)
            for chunk in cutlass.range_constexpr(topk // vector_words):
                chunk_route_begin = slab_route_begin + (Int32(chunk) ^ swizzle) * Int32(vector_words)
                for word in cutlass.range_constexpr(vector_words):
                    register_idx = token_slot * topk + chunk * vector_words + word
                    expert = slab_indices_i32[(chunk_route_begin + Int32(word)) * Int32(index_bytes // 4)]
                    is_invalid = (cutlass.Uint32(expert) >= cutlass.Uint32(self.expert_count)) | (not token_valid)
                    if is_invalid:
                        expert = sentinel_expert
                    expert_registers[register_idx] = expert

        return expert_registers, slab_scores

    @cute.jit
    def _sort_token_routes_and_scatter(
        self,
        expert_registers: cute.Tensor,
        within_expert_indices: cute.Tensor,
        score_slab: Optional[cute.Tensor],
        sorted_elements: cute.Tensor,
        expert_run_starts: cute.Tensor,
        topk_index_type: type,
    ) -> None:
        """Sort each token's routes by expert in registers (invalid routes sink to the end); a rank's routes form one
        run, so ``dst_order_id`` = position in the run and ``dst_token_cnt`` = run length. Scatter from the sorted
        registers (payload = within-expert index | physical topk slot << 16)."""
        topk = self.topk
        threads_per_cta = self.router_warps_per_cta * 32
        tokens_per_thread = self.router_tokens_per_thread
        experts_per_rank = self.experts_per_rank
        expert_count = Int32(self.expert_count)
        actual_token_count = Int32(self.max_tokens_per_rank)
        cta_token_begin = self._router_linear_cta_idx * Int32(tokens_per_thread * threads_per_cta)
        valid_rank_mask = self._device_workspace.tensor(self.valid_rank_mask_region)
        pairs = _odd_even_merge_sort_pairs(topk)
        for token_slot in cutlass.range_constexpr(tokens_per_thread):
            local_token = Int32(token_slot * threads_per_cta) + self._router_thread_idx
            token_idx = cta_token_begin + local_token
            keys = cute.make_rmem_tensor((topk,), cutlass.Int32)
            payload = cute.make_rmem_tensor((topk,), cutlass.Int32)
            for slot in cutlass.range_constexpr(topk):
                register_idx = token_slot * topk + slot
                _, physical_slot = self._router_value_coordinate(register_idx, topk_index_type)
                keys[slot] = expert_registers[register_idx]
                payload[slot] = within_expert_indices[register_idx] | (physical_slot << Int32(16))
            for low, high in pairs:
                key_low = keys[low]
                key_high = keys[high]
                swap_mask = Int32(0) - Int32(key_low > key_high)
                keys[low] = cutlass.min(key_low, key_high)
                keys[high] = cutlass.max(key_low, key_high)
                payload_low = payload[low]
                payload_high = payload[high]
                payload_delta = (payload_low ^ payload_high) & swap_mask
                payload[low] = payload_low ^ payload_delta
                payload[high] = payload_high ^ payload_delta
            # Forward pass: run start (first position of this rank) and the rank bit of every valid route.
            order = cute.make_rmem_tensor((topk,), cutlass.Int32)
            run_start = Int32(0)
            run_end_expert = Int32(0)
            block_mask = Int32(0)
            for position in cutlass.range_constexpr(topk):
                expert = keys[position]
                rank = expert // Int32(experts_per_rank)
                new_run = Int32(expert >= run_end_expert)
                run_start = run_start + (Int32(position) - run_start) * new_run
                run_end_expert = run_end_expert + ((rank + Int32(1)) * Int32(experts_per_rank) - run_end_expert) * new_run
                order[position] = Int32(position) - run_start
                valid = Int32(expert < expert_count)
                block_mask = block_mask | ((Int32(1) << rank) & (Int32(0) - valid))
            if token_idx < actual_token_count:
                valid_rank_mask[token_idx] = block_mask
            # Backward pass: run length, then the record scatter in sorted order.
            run_end = Int32(topk)
            for position in cutlass.range_constexpr(topk - 1, -1, -1):
                if cutlass.const_expr(position + 1 < topk):
                    starts_next = Int32(order[position + 1] == Int32(0))
                    run_end = run_end + (Int32(position + 1) - run_end) * starts_next
                expert = keys[position]
                valid = Int32(expert < expert_count)
                order_id = order[position] * valid
                copy_count = Int32(1) + (run_end - (Int32(position) - order[position]) - Int32(1)) * valid
                within_expert_index = payload[position] & Int32(0xFFFF)
                physical_slot = payload[position] >> Int32(16)
                flat_topk_index = token_idx * Int32(topk) + physical_slot
                copy_info = _CopyInfo(dst_order_id=order_id, dst_token_cnt=copy_count).pack()
                destination = expert_run_starts[expert] + within_expert_index
                score = None
                if cutlass.const_expr(self.carry_topk_scores):
                    score = score_slab[local_token * Int32(topk) + physical_slot]
                sorted_elements[destination] = _SortedElement(flat_topk_index, copy_info, score).pack()

    @cute.jit
    def _build_histogram(self, expert_registers: cute.Tensor, histogram: cute.Tensor) -> cute.Tensor:
        register_count = cute.size(expert_registers)
        within_expert_indices = cute.make_rmem_tensor((register_count,), cutlass.Int32)
        for register_idx in cutlass.range_constexpr(register_count):
            within_expert_indices[register_idx] = Int32(
                cute.arch.atomic_add(
                    histogram.iterator + expert_registers[register_idx], Int32(1), sem="relaxed", scope="cta"
                )
            )
        cute.arch.sync_threads()
        return within_expert_indices

    @cute.jit
    def _router_value_coordinate(self, register_idx: int, topk_index_type: type) -> Tuple[Int32, Int32]:
        """(token index, topk slot) of this thread's register ``register_idx`` under the whole-token ownership."""
        threads_per_cta = self.router_warps_per_cta * 32
        tokens_per_cta = self.router_tokens_per_thread * threads_per_cta
        token_slot = register_idx // self.topk
        token_idx = (
            self._router_linear_cta_idx * Int32(tokens_per_cta)
            + Int32(token_slot * threads_per_cta)
            + self._router_thread_idx
        )
        # Undo the slab-read swizzle: register slot s came from physical slot (chunk(s) XOR f(lane)) * V + s % V.
        slot = register_idx % self.topk
        vector_words = self.router_load_vector_words
        physical_slot = (Int32(slot // vector_words) ^ self._router_slot_swizzle()) * Int32(vector_words) + Int32(
            slot % vector_words
        )
        return (token_idx, physical_slot)

    @cute.jit
    def _route_metadata(self, flat_topk_index: Int32, copy_info: Int32) -> TokenSrcMetadata:
        """Unfold a sorted element (flat route index + packed _CopyInfo) into the wire metadata record."""
        info = _CopyInfo.unpack(copy_info)
        return TokenSrcMetadata(
            src_rank=Int32(self._router_local_rank),
            src_token=flat_topk_index // Int32(self.topk),
            src_topk=flat_topk_index % Int32(self.topk),
            dst_order_id=info.dst_order_id,
            dst_token_cnt=info.dst_token_cnt,
        )

    @cute.jit
    def _dump_contiguous_router_output(self, sorted_elements: cute.Tensor, total_valid_routes: Int32) -> None:
        block_thread_count = self.router_warps_per_cta * 32
        metadata_address = self._device_workspace.ptr(self.sorted_metadata_region).toint()
        if cutlass.const_expr(self.carry_topk_scores):
            score_address = self._device_workspace.ptr(self.sorted_scores_region).toint()
        dump_round_count = (total_valid_routes + Int32(block_thread_count - 1)) // Int32(block_thread_count)
        for dump_round in cutlass.range(dump_round_count, unroll=4):
            position = Int32(dump_round * block_thread_count) + self._router_thread_idx
            predicate = Int32(position < total_valid_routes)
            element = _SortedElement.from_packed(sorted_elements[position])
            metadata = self._route_metadata(element.flat_topk_index, element.copy_info)
            stg_b64(metadata_address + Int64(position) * Int64(TokenSrcMetadata.nbytes), metadata.pack(), predicate)
            if cutlass.const_expr(self.carry_topk_scores):
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
        if cutlass.const_expr(self.carry_topk_scores):
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
                    element = _SortedElement.from_packed(sorted_elements[predicate * (run_begin + route)])
                    output_position = dump_begin + route
                    metadata = self._route_metadata(element.flat_topk_index, element.copy_info)
                    packed_metadata = metadata.pack()
                    stg_b64(
                        metadata_address + Int64(output_position) * Int64(TokenSrcMetadata.nbytes),
                        packed_metadata,
                        predicate,
                    )
                    if cutlass.const_expr(self.carry_topk_scores):
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
            while cute.arch.load(sizes_ready, Int32, sem="acquire", scope="sys") != Int32(self.world_size):
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
    def valid_rank_mask_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.valid_rank_mask_region)

    @cute.jit
    def fc1_topk_scores_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.carry_topk_scores):
            return None
        return device_workspace.tensor(self.fc1_topk_scores_region)


class TokenCommDedup(KernelComponent):
    """Public fused-kernel token communication component."""

    transfer_warp_count: ClassVar[int] = 4
    transfer_thread_count: ClassVar[int] = transfer_warp_count * 32
    # Order-buffer hidden chunk (= stage slot); 1 KB chunks measured 0.1 ms slower at DSV4.
    standalone_chunk_bytes: ClassVar[int] = 2048
    reduction_round_group: ClassVar[int] = 4
    # Last-copy reduce: two copies per 1-D TMA through two buffers, so per-warp SMEM is 2 x 2 x chunk.
    token_back_copies_per_fill: ClassVar[int] = 2
    token_back_fill_buffers: ClassVar[int] = 2
    # Combine-warp fc2_done prefix-wait spin cadence (ns). Diagnostic override: MEGA_DEDUP_WAIT_SLEEP_NS.
    fc2_done_wait_sleep_ns: ClassVar[int] = int(os.environ.get("MEGA_DEDUP_WAIT_SLEEP_NS", "500"))
    # Backoff cap (ns) of that spin; polling every 0.5 us slows the MMA/epilogue warps (DSV4 T8192: -12 us).
    fc2_done_wait_sleep_max_ns: ClassVar[int] = int(os.environ.get("MEGA_DEDUP_WAIT_SLEEP_MAX_NS", "8000"))
    # SMEM cap per combine warp for the fill buffers (reuse mode shares the token_in overlay).
    stage_ring_budget_bytes: ClassVar[int] = 12288
    transfer_lifetime_barrier_id: ClassVar[int] = 9
    grid_sync_barrier_id: ClassVar[int] = 10
    standalone_size_barrier_id: ClassVar[int] = 11
    token_in_size_barrier_id: ClassVar[int] = 12

    fc1_ready_region = "nvlink.token_comm_dedup.fc1_ready"
    fc1_activation_region = "nvlink.token_comm_dedup.fc1_activation"
    fc1_activation_sf_region = "nvlink.token_comm_dedup.fc1_activation_sf"
    fc2_done_region = "nvlink.token_comm_dedup.fc2_done"
    # Order buffer the FC2 epilogue scatters into at [src_token, src_rank, dst_order_id].
    fc2_order_output_region = "nvlink.token_comm_dedup.fc2_order_output"
    pre_reduced_activation_region = "nvlink.token_comm_dedup.pre_reduced_activation"
    pre_reduced_activation_sf_region = "nvlink.token_comm_dedup.pre_reduced_activation_sf"
    token_back_schedule_region = "nvlink.token_comm_dedup.token_back_schedule"  # nosec B105
    # Rank-major copies pulled by a token's first same-rank occurrence (dispatch dedup).

    token_in_mbarrier_region = "nvlink.token_comm_dedup.main_smem.token_in_mbarriers"  # nosec B105
    token_back_mbarrier_region = "nvlink.token_comm_dedup.main_smem.token_back_mbarriers"  # nosec B105
    expert_sizes_smem_region = "nvlink.token_comm_dedup.main_smem.expert_sizes"
    token_in_activation_smem_region = "nvlink.token_comm_dedup.main_smem.token_in_activation"  # nosec B105
    token_in_sf_smem_region = "nvlink.token_comm_dedup.main_smem.token_in_sf"  # nosec B105
    token_back_activation_smem_region = "nvlink.token_comm_dedup.main_smem.token_back_activation"  # nosec B105
    # hybrid_warps: second set of rings for the dispatch warps, overlaid on their token_in stage.
    token_back_reuse_activation_smem_region = "nvlink.token_comm_dedup.main_smem.token_back_reuse_activation"  # nosec B105
    token_back_sf_smem_region = "nvlink.token_comm_dedup.main_smem.token_back_sf"  # nosec B105

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
        self.carry_topk_scores = True

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
        self._router = _DedupMetadataPushRouter(problem_desc, impl_desc)
        self._nvlink_barrier = NvlinkBarrier(world_size=self.world_size, barrier_id=self.grid_sync_barrier_id)
        self._device_workspace = None
        self._token_comm_args = None
        self._local_rank = None
        self._linear_cta_idx = None
        self._transfer_warp_idx = None
        self._lane_idx = None
        self._transfer_group = None

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
            raise ValueError(f"topk must not exceed expert_count, got {self.topk} and {self.expert_count}.")
        if self.token_back_mode not in ("standalone_warps", "reuse_dispatch_warps", "hybrid_warps"):
            raise ValueError(
                f"Combine dedup pushes the reduced rows from the combine warps; token_back_mode {self.token_back_mode!r} "
                "is not supported (use standalone_warps, reuse_dispatch_warps or hybrid_warps)."
            )
        if self.token_back_schedule_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"token_back_schedule_mode must be static or atomic_counter, got {self.token_back_schedule_mode!r}."
            )
        if not 1 <= self.token_in_flag_batch <= 32:
            raise ValueError(f"token_in_flag_batch must be in [1, 32], got {self.token_in_flag_batch}.")
        if self.tokens_per_fc1_ready_slot % self.token_padding_block != 0:
            raise ValueError("tokens_per_fc1_ready_slot must be divisible by token_padding_block.")
        if self.token_back_enabled and self.fc2_done_signals_per_token_tile <= 0:
            raise ValueError("fc2_done_signals_per_token_tile must be positive when token-back is enabled.")
        if self.combine_format.is_quantized or self.combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError("Combine dedup supports the BF16 combine format only.")
        if self.hidden_size % 128 != 0:
            raise ValueError("Combine dedup requires hidden_size divisible by the 128-wide FC2 epilogue tile.")
        element_block = self.activation_sf_vector_size * 4
        if self.hidden_size % element_block != 0:
            raise ValueError(f"{self.quant_kind} requires hidden_size divisible by {element_block}.")
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
        # Dedup always pushes from the combine warps (reuse_dispatch_warps / standalone_warps).
        return True

    @property
    def output_token_bytes(self) -> int:
        return self.hidden_size * int(self.combine_format.act_dtype.width) // 8

    # The FC2 epilogue writes 128-hidden CTA tiles; a chunk must hold whole tiles so one tile never straddles two.
    order_chunk_granule_bytes: ClassVar[int] = 256

    @property
    def token_back_stage_slots(self) -> int:
        """Chunk slots of SMEM per combine warp: fill buffers x copies per fill."""
        return self.token_back_fill_buffers * self.token_back_copies_per_fill

    def token_back_chunk_bytes(self, available_bytes_per_warp: int) -> int:
        """Bytes per hidden chunk of the order buffer / last-copy reduce: the per-warp budget divided by the
        ``token_back_stage_slots`` chunk slots, capped at the standalone chunk and the row, rounded down to the 256 B
        granule; the row's last chunk may be partial (padded in the buffer)."""
        row_bytes = self.output_token_bytes
        granule = self.order_chunk_granule_bytes
        chunk = min(self.standalone_chunk_bytes, available_bytes_per_warp // self.token_back_stage_slots)
        chunk = min(chunk, ceil_div(row_bytes, granule) * granule)
        chunk = chunk // granule * granule
        if chunk < granule:
            raise ValueError(
                f"Token-back fill buffers too small: {available_bytes_per_warp} B per warp for "
                f"{self.token_back_stage_slots} chunk slots gives a {chunk} B chunk (need >= {granule} B)."
            )
        if chunk > 32 * 16 * self.reduction_round_group:
            raise ValueError(f"Token-back chunk {chunk} B exceeds one reduce round; its accumulators must fit registers.")
        return chunk

    @property
    def hybrid_token_back(self) -> bool:
        """hybrid_warps: the standalone combine warps AND the dispatch warps (after token_in) both run token_back.
        Twice the combine workers for the same chunk size; wins where the combine tail dominates (Qwen T>=8k:
        -5..-7 %), loses where mainloop interference dominates (DSV4 8k: +6 %)."""
        return self.token_back_mode == "hybrid_warps"  # nosec B105

    @property
    def token_back_worker_groups(self) -> int:
        return 2 if self.hybrid_token_back else 1

    def _token_back_available_bytes_per_warp(self) -> int:
        if self.token_back_mode in ("standalone_warps", "hybrid_warps"):
            # hybrid: one chunk size for both groups (the order-buffer layout is shared).
            return self.stage_ring_budget_bytes
        activation_bytes = self.bytes_per_token
        sf_bytes = self.activation_sf_hidden_padded * int(self.activation_sf_dtype.width) // 8
        return max(activation_bytes + sf_bytes, self.stage_ring_budget_bytes)

    @property
    def order_chunk_bytes(self) -> int:
        """Hidden chunk of the order buffer layout [T, R, chunks_per_row, K, chunk]."""
        return self.token_back_chunk_bytes(self._token_back_available_bytes_per_warp())

    @property
    def order_chunks_per_row(self) -> int:
        return ceil_div(self.output_token_bytes, self.order_chunk_bytes)

    @property
    def order_block_bytes(self) -> int:
        """Bytes of one (src_token, src_rank) block: chunks_per_row x K x chunk."""
        return self.order_chunks_per_row * self.topk * self.order_chunk_bytes

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
        return self._router.worst_case_padded_tokens(self.tokens_per_fc1_ready_slot) // self.tokens_per_fc1_ready_slot

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
        self._device_workspace = device_workspace
        self._token_comm_args = token_comm_args
        self._local_rank = local_rank
        self._linear_cta_idx = linear_cta_idx
        thread_idx, _, _ = cute.arch.thread_idx()
        transfer_thread_idx = thread_idx % Int32(self.transfer_thread_count)
        self._transfer_warp_idx = cute.arch.make_warp_uniform(transfer_thread_idx // Int32(32))
        self._lane_idx = transfer_thread_idx % Int32(32)
        # Transfer-warp group: 0 = dispatch warps (8-11), 1 = standalone combine warps (12-15).
        self._transfer_group = cute.arch.make_warp_uniform(
            Int32(thread_idx // Int32(self.transfer_thread_count) == Int32(3))
        )
        self._nvlink_barrier.assign_device_members(device_workspace, token_comm_args.peer_rank_ptr_mapper)

    def remove_device_members(self) -> None:
        self._nvlink_barrier.remove_device_members()
        self._device_workspace = None
        self._token_comm_args = None
        self._local_rank = None
        self._linear_cta_idx = None
        self._transfer_warp_idx = None
        self._lane_idx = None
        self._transfer_group = None

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "TokenCommDedup":
        if values:
            raise ValueError("TokenCommDedup carries no MLIR values.")
        return self

    def _register_main_workspace(self, workspace: DeviceWorkspace) -> None:
        """Register the fused kernel's GMEM regions: FC1 pools, the token-back machinery (``fc2_done`` + order
        buffer) and, unless ``reduce_topk_in_kernel``, the symmetric (T, R, H) combine plane (never cleared: every
        exposed cell is rewritten each launch)."""
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
        activation_sf_element_count = self.worst_case_sf_token_count * self.activation_sf_hidden_padded
        workspace.register(
            self.fc1_activation_sf_region,
            self.activation_sf_dtype,
            (activation_sf_element_count,),
            buffer_space="local",
            byte_alignment=128,
        )

        workspace.register(
            self.fc2_done_region, cutlass.Int32, (self.experts_per_rank,), buffer_space="local", reset="tail_reset"
        )
        # Order buffer [T, R, chunks_per_row, K, chunk]; never cleared (every byte read was written this launch).
        order_bytes = self.max_tokens_per_rank * self.world_size * self.order_block_bytes
        workspace.register(
            self.fc2_order_output_region,
            self.combine_format.act_dtype,
            (order_bytes * 8 // int(self.combine_format.act_dtype.width),),
            buffer_space="local",
            byte_alignment=128,
        )
        if self.token_back_schedule_mode == "atomic_counter":  # nosec B105
            workspace.register(
                self.token_back_schedule_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
            )

        if not self.reduce_topk_in_kernel:
            # One slot per pushing rank (not per topk): the receiver sums the slots named by valid_rank_mask.
            workspace.register(
                self.pre_reduced_activation_region,
                self.combine_format.act_dtype,
                (self.max_tokens_per_rank, self.world_size, self.hidden_size),
                buffer_space="shared",
                mem_order=(2, 1, 0),
                byte_alignment=128,
            )

    def register_smem_regions(self, workspace: SmemWorkspace) -> None:
        workspace.register_mbarrier(self.token_in_mbarrier_region, self.transfer_warp_count)
        if self.token_back_enabled:
            workspace.register_mbarrier(
                self.token_back_mbarrier_region,
                self.transfer_warp_count * self.token_back_worker_groups * self.token_back_fill_buffers,
            )
        workspace.register_tensor(
            self.expert_sizes_smem_region, cutlass.Int32, (self.experts_per_rank,), byte_alignment=16
        )
        transfer_overlay = workspace.create_overlay("nvlink.token_comm_dedup.main_smem.transfer")
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
            (self.transfer_warp_count, (self.activation_sf_vector_size, self.activation_sf_hidden_padded)),
            stride=(self.activation_sf_hidden_padded, (0, 1)),
            byte_alignment=16,
        )
        if not self.token_back_enabled:
            return

        if self.token_back_mode in ("standalone_warps", "hybrid_warps"):
            token_back_lifetime = workspace.create_overlay(
                "nvlink.token_comm_dedup.main_smem.standalone_token_back"
            ).add_lifetime("token_back")
        else:
            token_back_lifetime = transfer_overlay.add_lifetime("token_back")

        if self.token_back_push_data:
            # Per warp: two fill buffers of two chunk slots; reuse mode shares the token_in overlay.
            chunk_bytes = self.order_chunk_bytes
            chunk_elements = chunk_bytes * 8 // int(self.combine_format.act_dtype.width)
            token_back_lifetime.register_tensor(
                self.token_back_activation_smem_region,
                self.combine_format.act_dtype,
                (self.transfer_warp_count, self.token_back_stage_slots, chunk_elements),
                byte_alignment=16,
            )
            if self.hybrid_token_back:
                # Dispatch warps' buffers live in their own token_in overlay (free once token_in is done).
                transfer_overlay.add_lifetime("token_back").register_tensor(
                    self.token_back_reuse_activation_smem_region,
                    self.combine_format.act_dtype,
                    (self.transfer_warp_count, self.token_back_stage_slots, chunk_elements),
                    byte_alignment=16,
                )
        if False:  # bf16-only combine: no scale stage
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

    @cute.jit
    def fc1_activation_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return cute.make_tensor(
            device_workspace.ptr(self.fc1_activation_region),
            cute.make_layout((self.worst_case_token_count, self.hidden_size), stride=(self.hidden_size, 1)),
        )

    @cute.jit
    def fc1_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
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
        """Base of the order buffer as a (T*R*K, 1, hidden) tensor. Only the base pointer and the hidden extent
        (shape[2]) are meaningful: the epilogue addresses rows through the chunked layout
        [T, R, chunks_per_row, K, chunk] (``dedup_chunk_bytes`` / ``dedup_chunks_per_row`` in its impl_desc)."""
        rows = self.max_tokens_per_rank * self.world_size * self.topk
        return cute.make_tensor(
            device_workspace.ptr(self.fc2_order_output_region),
            cute.make_layout((rows, 1, self.hidden_size), stride=(self.hidden_size, self.hidden_size, 1)),
        )

    @cute.jit
    def valid_rank_mask_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return self._router.valid_rank_mask_tensor(device_workspace)
    @cute.jit
    def fc2_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        return None
    @cute.jit
    def pre_reduced_activation_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        """The (tokens, world_size, hidden) rank-slot staging plane, or None under in-kernel top-k reduction."""
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            return None
        return device_workspace.tensor(self.pre_reduced_activation_region)

    @cute.jit
    def pre_reduced_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        return None
    @cute.jit
    def token_in(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """Wait for pushed metadata, then pull activation payloads into local pools (same as the non-dedup dispatch:
        every copy pulls its row from the source rank; dedup happens on the combine side only)."""
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
            (mark_alignment(source_sizes, 16) if cutlass.const_expr(copy_elements == 4) else source_sizes),
            (copy_elements,),
        )
        destination_size_vectors = cute.zipped_divide(owned_sizes, (copy_elements,))
        size_vector_count = cute.size(destination_size_vectors, mode=[1])
        size_copy_atom = _copy_atom(cutlass.Int32, copy_elements * 32)
        size_copy_rounds = ceil_div(size_vector_count, self.transfer_thread_count)
        size_copy_registers = cute.make_rmem_tensor((copy_elements, size_copy_rounds), cutlass.Int32)
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
        if cutlass.const_expr(self.token_back_mode in ("standalone_warps", "hybrid_warps")):
            sizes_ready_barrier = pipeline.NamedBarrier(
                barrier_id=self.standalone_size_barrier_id, num_threads=2 * self.transfer_thread_count
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
        activation_bytes = cute.cosize(warp_activation_stage) * int(self.activation_dtype.width) // 8
        activation_sf_bytes = cute.cosize(warp_sf_stage) * int(self.activation_sf_dtype.width) // 8
        sf_copy_elements = 4
        source_sf_values = cute.slice_(warp_sf_stage, (0, None))
        source_sf_vectors = cute.zipped_divide(source_sf_values, (sf_copy_elements,))
        sf_copy_atom = _copy_atom(self.activation_sf_dtype, sf_copy_elements * int(self.activation_sf_dtype.width))

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
                pull_count = (expert_valid_end - next_dense_token + global_warp_count - Int32(1)) // global_warp_count

            for pull_round in cutlass.range(pull_count, unroll=1):
                dense_token_idx = next_dense_token + Int32(pull_round) * global_warp_count
                token_in_expert = dense_token_idx - expert_valid_begin
                pool_token_idx = pool_expert_bases[local_expert] + token_in_expert
                sf_token_idx = expert_sf_begin + token_in_expert
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

                cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.sync_warp()
                ready_address = (fc1_ready_counter + ready_slot_idx).toint()
                flag_tracker = flag_tracker.accumulate(Int32(0), self.token_in_flag_batch, ready_address)
                cute.arch.sync_warp()
                pull_phase = pull_phase ^ Int32(1)

            next_dense_token = next_dense_token + pull_count * global_warp_count
            expert_valid_begin = expert_valid_end
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
        if cutlass.const_expr(
            self.token_back_enabled and self.token_back_mode in ("reuse_dispatch_warps", "hybrid_warps")
        ):
            iket.range_push("token_in.transfer_barrier")
            transfer_lifetime_barrier = pipeline.NamedBarrier(
                barrier_id=self.transfer_lifetime_barrier_id, num_threads=self.transfer_thread_count
            )
            transfer_lifetime_barrier.arrive_and_wait()
            iket.range_pop()

    @cute.jit
    def token_back(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """Combine warps: walk the pool metadata in expert order, wait for the ``fc2_done`` prefix up to the current
        expert, and for every last copy (``order == cnt - 1``) reduce the block's copies in fp32 (order 0..cnt-1),
        round once to bf16 and push it to the source rank. Single-copy rows are pushed by the FC2 epilogue."""
        transfer_warp_idx = self._transfer_warp_idx
        lane_idx = self._lane_idx
        if cutlass.const_expr(not self.token_back_enabled):
            return

        workers_per_cta = self.transfer_warp_count * self.token_back_worker_groups
        worker_in_cta = transfer_warp_idx
        if cutlass.const_expr(self.hybrid_token_back):
            worker_in_cta = self._transfer_group * Int32(self.transfer_warp_count) + transfer_warp_idx
        global_worker_idx = self._linear_cta_idx * Int32(workers_per_cta) + worker_in_cta
        global_worker_count = Int32(self.promised_launchable_sm_count * workers_per_cta)
        token_back_mbarriers = smem_workspace.ptr(self.token_back_mbarrier_region, smem_base)
        worker_mbarriers = token_back_mbarriers + worker_in_cta * Int32(self.token_back_fill_buffers)
        token_back_stages = smem_workspace.tensor(self.token_back_activation_smem_region, smem_base)
        worker_stages = token_back_stages[transfer_warp_idx, None, None]  # (stage_slots, chunk_elements)
        if cutlass.const_expr(self.hybrid_token_back):
            # Dispatch warps (group 0) reduce in the rings overlaid on their finished token_in stage.
            reuse_stages = smem_workspace.tensor(self.token_back_reuse_activation_smem_region, smem_base)
            if self._transfer_group == Int32(0):
                worker_stages = reuse_stages[transfer_warp_idx, None, None]
        chunk_bytes = cute.size(worker_stages, mode=[1]) * int(self.combine_format.act_dtype.width) // 8
        if lane_idx == Int32(0):
            for buffer_idx in cutlass.range_constexpr(self.token_back_fill_buffers):
                cute.arch.mbarrier_init(worker_mbarriers + buffer_idx, 1)
        cute.arch.sync_warp()

        owned_sizes = smem_workspace.tensor(self.expert_sizes_smem_region, smem_base)
        if cutlass.const_expr(self.token_back_mode in ("standalone_warps", "hybrid_warps")):
            # The dispatch warps already met this barrier inside token_in (hybrid); only the standalone group waits.
            if cutlass.const_expr(self.token_back_mode == "standalone_warps") or self._transfer_group == Int32(1):  # nosec B105
                sizes_ready_barrier = pipeline.NamedBarrier(
                    barrier_id=self.standalone_size_barrier_id, num_threads=2 * self.transfer_thread_count
                )
                sizes_ready_barrier.arrive_and_wait()

        pool_expert_bases = self._router.pool_expert_base_tensor(self._device_workspace)
        token_metadata_pointer = self._router.token_src_metadata_pointer(self._device_workspace)
        fc2_done = self._device_workspace.ptr(self.fc2_done_region)
        order_pointer = self._device_workspace.ptr(self.fc2_order_output_region)
        output_token_bytes = self.output_token_bytes
        # Fills issued by this worker so far: fill s uses buffer s % 2 and phase (s // 2) & 1 of its mbarrier.
        fill_sequence = Int32(0)

        iket.range_push("token_back.work")
        next_dense_token = global_worker_idx - global_worker_count
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):  # nosec B105
            next_dense_token = Int32(0)
        next_dense_token = self.next_token(next_dense_token)
        completed_expert_prefix = Int32(0)
        expert_valid_begin = Int32(0)
        local_expert = Int32(0)
        while local_expert < Int32(self.experts_per_rank):
            expert_token_count = owned_sizes[local_expert]
            expert_valid_end = expert_valid_begin + expert_token_count
            while next_dense_token < expert_valid_end:
                token_in_expert = next_dense_token - expert_valid_begin
                pool_token_idx = pool_expert_bases[local_expert] + token_in_expert
                metadata = TokenSrcMetadata.load(
                    token_metadata_pointer.toint() + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                )
                # Single-copy rows were pushed straight from the FC2 epilogue; only multi-copy blocks are reduced here.
                if (metadata.is_last_copy != Int32(0)) and (metadata.dst_token_cnt > Int32(1)):
                    iket.range_push("token_back.wait_fc2_warp_prefix")
                    wait_sleep_ns = Int32(self.fc2_done_wait_sleep_ns)
                    while completed_expert_prefix <= local_expert:
                        prefix_base = (completed_expert_prefix // Int32(32)) * Int32(32)
                        checked_expert = prefix_base + Int32(31) - lane_idx
                        lane_ready = checked_expert >= Int32(self.experts_per_rank)
                        if checked_expert < Int32(self.experts_per_rank):
                            checked_count = owned_sizes[checked_expert]
                            checked_target = ((checked_count + Int32(self.tokens_per_fc1_ready_slot - 1))
                                // Int32(self.tokens_per_fc1_ready_slot)) * Int32(self.fc2_done_signals_per_token_tile)
                            lane_ready = cute.arch.load(fc2_done + checked_expert, Int32,
                                                        sem="acquire", scope="gpu") >= checked_target
                        ready_mask = cute.arch.vote_ballot_sync(lane_ready)
                        completed_expert_prefix = cutlass.min(
                            prefix_base + Int32(cute.arch.clz(~ready_mask)), Int32(self.experts_per_rank))
                        if completed_expert_prefix <= local_expert:
                            nanosleep(wait_sleep_ns)
                            wait_sleep_ns = cutlass.min(wait_sleep_ns * Int32(2), Int32(self.fc2_done_wait_sleep_max_ns))
                    # Order every lane's fc2_done acquire before the TMA that reads the completed copies.
                    cute.arch.sync_warp()
                    iket.range_pop()
                    block_index = Int64(metadata.src_token) * Int64(self.world_size) + Int64(metadata.src_rank)
                    block_address = order_pointer.toint() + block_index * Int64(self.order_block_bytes)
                    fill_sequence = self._reduce_and_push(
                        block_address,
                        metadata.dst_token_cnt,
                        metadata.src_rank,
                        metadata.src_token,
                        worker_mbarriers,
                        worker_stages,
                        fill_sequence,
                        chunk_bytes,
                    )
                next_dense_token = self.next_token(next_dense_token)

            expert_valid_begin = expert_valid_end
            local_expert = local_expert + Int32(1)
        iket.range_pop()

    @cute.jit
    def _issue_fill(
        self,
        sequence: Int32,
        fill_idx: Int32,
        fills_per_chunk: Int32,
        copy_count: Int32,
        worker_mbarriers: cute.Pointer,
        stage_base,
        block_address: Int64,
        chunk_bytes: int,
    ) -> None:
        """Elected lane: one 1-D TMA of the two copies of fill ``fill_idx`` (chunk-major, copies ascending) into
        buffer ``sequence % 2``, arming that buffer's mbarrier with the byte count."""
        chunk_idx = fill_idx // fills_per_chunk
        first_copy = (fill_idx - chunk_idx * fills_per_chunk) * Int32(self.token_back_copies_per_fill)
        copies = cutlass.min(Int32(self.token_back_copies_per_fill), copy_count - first_copy)
        buffer_idx = sequence % Int32(self.token_back_fill_buffers)
        mbarrier = worker_mbarriers + buffer_idx
        destination = cute.make_ptr(
            cutlass.Uint8,
            stage_base + buffer_idx * Int32(self.token_back_copies_per_fill * chunk_bytes),
            AddressSpace.smem,
            assumed_align=16,
        )
        source = block_address + (Int64(chunk_idx) * Int64(self.topk) + Int64(first_copy)) * Int64(chunk_bytes)
        cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, copies * Int32(chunk_bytes))
        tma_load_1d(destination, source, mbarrier, copies * Int32(chunk_bytes))

    @cute.jit
    def _reduce_and_push(
        self,
        block_address: Int64,
        copy_count: Int32,
        src_rank: Int32,
        src_token: Int32,
        worker_mbarriers: cute.Pointer,
        worker_stages: cute.Tensor,
        fill_sequence: Int32,
        chunk_bytes: int,
    ) -> Int32:
        """Reduce the ``copy_count`` copies of the order block at ``block_address`` chunk by chunk (two copies per
        TMA through two buffers, fp32 accumulation in order 0..cnt-1, one bf16 rounding) and push each chunk.
        Returns the advanced fill sequence."""
        lane_idx = self._lane_idx
        copies_per_fill = self.token_back_copies_per_fill
        output_token_bytes = self.output_token_bytes
        chunk_count = ceil_div(output_token_bytes, chunk_bytes)
        round_group = min(self.reduction_round_group, ceil_div(chunk_bytes // 16, 32))
        destination_slot = Int32(self._local_rank)
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            destination_slot = Int32(0)
        peer_offset = self._token_comm_args.peer_rank_ptr_mapper.map(Int64(0), src_rank, Int64(0))
        remote_activation = cute.make_tensor(
            cute.make_ptr(
                self._token_comm_args.pre_reduced_activation.dtype,
                self._token_comm_args.pre_reduced_activation.iterator.toint() + peer_offset,
                AddressSpace.gmem,
                assumed_align=(self._token_comm_args.pre_reduced_activation.iterator.max_alignment),
            ),
            self._token_comm_args.pre_reduced_activation.layout,
        )
        destination_row = remote_activation[Int64(src_token), destination_slot, None]
        word_atom = _copy_atom(cutlass.Int32, 128)
        vector_layout = cute.make_layout((4,))
        stage_base = worker_stages.iterator.toint()
        one2 = f32x2_pack(cutlass.Float32(1.0), cutlass.Float32(1.0))  # {1.0, 1.0} weight for the FFMA2 adds
        fills_per_chunk = (copy_count + Int32(copies_per_fill - 1)) // Int32(copies_per_fill)
        fill_count = fills_per_chunk * Int32(chunk_count)

        iket.range_push("token_back.push_data")
        with cute.arch.elect_one():
            self._issue_fill(fill_sequence, Int32(0), fills_per_chunk, copy_count, worker_mbarriers, stage_base, block_address, chunk_bytes)
        # Whole-chunk fp32 accumulators (round_group x 32 lanes x 16 B covers the chunk, checked at construction).
        words = cute.make_rmem_tensor((4,), cutlass.Int32)
        acc = cute.make_rmem_tensor((4, round_group), cutlass.Int64)
        # Rolled loop over the fills of the token: the body exists once in SASS (icache footprint).
        for fill_idx in cutlass.range(fill_count, unroll=1):
            chunk_idx = fill_idx // fills_per_chunk
            fill_in_chunk = fill_idx - chunk_idx * fills_per_chunk
            first_copy = fill_in_chunk * Int32(copies_per_fill)
            sequence = fill_sequence + fill_idx
            buffer_base = stage_base + (sequence % Int32(self.token_back_fill_buffers)) * Int32(copies_per_fill * chunk_bytes)
            current_vectors = cutlass.min(Int32(chunk_bytes), Int32(output_token_bytes) - chunk_idx * Int32(chunk_bytes)) // Int32(16)
            # Prefetch fill i+1 into the other buffer. If that buffer just fed a bulk copy, wait for its READ only.
            if fill_idx + Int32(1) < fill_count:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                with cute.arch.elect_one():
                    self._issue_fill(sequence + Int32(1), fill_idx + Int32(1), fills_per_chunk, copy_count,
                                     worker_mbarriers, stage_base, block_address, chunk_bytes)
            if fill_in_chunk == Int32(0):
                for group in cutlass.range_constexpr(round_group):
                    for word in cutlass.range_constexpr(4):
                        acc[word, group] = Int64(0)
            cute.arch.mbarrier_wait(
                worker_mbarriers + sequence % Int32(self.token_back_fill_buffers),
                (sequence // Int32(self.token_back_fill_buffers)) & Int32(1),
            )
            iket.range_push("token_back.reduce")
            # Copies accumulate in 0..cnt-1 order; groups are 512 B apart so adjacent lanes keep contiguous 16 B.
            for copy_in_fill in cutlass.range_constexpr(copies_per_fill):
                if first_copy + Int32(copy_in_fill) < copy_count:
                    for group in cutlass.range_constexpr(round_group):
                        vector_idx = Int32(group * 32) + lane_idx
                        if vector_idx < current_vectors:
                            slot_vector = cute.make_tensor(
                                cute.make_ptr(cutlass.Int32, buffer_base + Int32(copy_in_fill * chunk_bytes) + vector_idx * Int32(16),
                                              AddressSpace.smem, assumed_align=16),
                                vector_layout,
                            )
                            cute.copy(word_atom, slot_vector, words)
                            for word in cutlass.range_constexpr(4):
                                acc[word, group] = fma_rn_f32x2_bf16x2(words[word], one2, acc[word, group])
            iket.range_pop()
            if fill_in_chunk == fills_per_chunk - Int32(1):
                # Round once to bf16 into slot 0 of this buffer and push the chunk from there.
                for group in cutlass.range_constexpr(round_group):
                    vector_idx = Int32(group * 32) + lane_idx
                    if vector_idx < current_vectors:
                        slot0_vector = cute.make_tensor(
                            cute.make_ptr(cutlass.Int32, buffer_base + vector_idx * Int32(16), AddressSpace.smem, assumed_align=16),
                            vector_layout,
                        )
                        for word in cutlass.range_constexpr(4):
                            words[word] = cvt_rn_bf16x2_f32x2(acc[word, group])
                        cute.copy(word_atom, words, slot0_vector)
                # Generic-proxy SMEM writes must be visible to the bulk copy (async proxy) that reads slot 0.
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_warp()
                chunk_source = cute.make_ptr(cutlass.Uint8, buffer_base, AddressSpace.smem, assumed_align=16)
                destination_chunk = cute.make_ptr(
                    cutlass.Uint8,
                    destination_row.iterator.toint() + Int64(chunk_idx) * Int64(chunk_bytes),
                    AddressSpace.gmem,
                    assumed_align=16,
                )
                with cute.arch.elect_one():
                    if cutlass.const_expr(self.reduce_topk_in_kernel):
                        cp_reduce_async_bulk_add_bf16_s2g(destination_chunk, chunk_source, current_vectors * Int32(16))
                    else:
                        cp_async_bulk_s2g(destination_chunk, chunk_source, current_vectors * Int32(16))
                cute.arch.cp_async_bulk_commit_group()
            cute.arch.sync_warp()
        # The pushes of this token must have landed before the worker moves on (matches the previous per-chunk wait).
        cute.arch.cp_async_bulk_wait_group(0)
        iket.range_pop()
        cute.arch.sync_warp()
        return fill_sequence + fill_count

    @cute.jit
    def next_token(self, current_token: Int32) -> Int32:
        # Static stride over every token_back worker in the grid; must match ``global_worker_count`` in ``token_back``.
        workers_per_cta = self.transfer_warp_count * self.token_back_worker_groups
        global_worker_count = self.promised_launchable_sm_count * workers_per_cta
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


__all__ = ["TokenCommDedup"]
