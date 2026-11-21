# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-round push dispatch for small buckets of the pull-dispatch MegaMoE (SM100 / SM103).

``_MetadataPushRouter`` pays three cross-rank round trips before FC1 can read a token: a size broadcast, a
metadata push, and the activation pull that ``token_in`` issues once the metadata has landed. At a few tokens per
rank every one of them is latency, not bytes. This component replaces all three with one:

  push       every source CTA walks a slice of this rank's tokens. For each (token, top-k slot) whose expert is
             valid it writes one record into the owning rank's inbox, and for each distinct owning rank it writes
             the token's NVFP4 row and scale row once into that rank's per-source plane. Then one system fence and
             one counter per (source CTA, destination), whose value is the record count + 1.
  sort       CTA 0 of the destination waits until every (source, CTA) counter is non-zero, counting-sorts the
             records it received by local expert, and publishes the same outputs the metadata-push router
             publishes: local expert sizes, padded pool bases, ``TokenSrcMetadata`` and FC1 top-k scores per pool
             row, then the sizes-ready and metadata-ready flags.

``token_in`` keeps its loop and its pool / scale-factor layout and only reads each row from the local plane instead
of the peer's staged activation, so FC1 sees the same bytes and every result is unchanged. The scheduler, FC1 / FC2,
token-back and the tail are untouched.

The inbox counts need no reset between calls: one producer per word, tagged with the per-workspace call epoch every
rank advances once per call, in two banks by epoch parity, so a word belongs to call N exactly when its epoch field
is N's. A source pushes call N + 1 only after its call-N kernel passed the drain barrier, which the destination
reaches after it finished reading call N's inbox and plane.
"""

from typing import ClassVar, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op

from ...api import ImplDesc, KernelComponent, OptionalRequirement, ProblemDesc
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.dsl_helpers import smem_exclusive_prefix
from ...helpers.ptx_helpers import nanosleep, stg_b64, stg_f32
from ...helpers.smem_workspace import SmemWorkspace
from ...helpers.utils import ceil_div, round_up
from ...quant_def import QuantKind
from ..token_protocol import TokenSrcMetadata
from .symmetric_buffer import SymmetricBufferDevice


@dsl_user_op
def _ldg_v4_b32(address: Int64, *, loc=None, ip=None):
    """Coherent 16-byte global load (never ``.nc``: the source rows are written by the preceding launch)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value()],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), result, [index])) for index in range(4))


@dsl_user_op
def _stg_v4_b32(address: Int64, v0: Int32, v1: Int32, v2: Int32, v3: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [address.ir_value(), v0.ir_value(), v1.ir_value(), v2.ir_value(), v3.ir_value()],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_b64(address: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [address.ir_value()],
            "ld.global.b64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_acquire_sys_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value()],
            "ld.acquire.sys.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_as_s32(value: cutlass.Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(), [value.ir_value()], "mov.b32 $0, $1;", "=r,f", asm_dialect=0, loc=loc, ip=ip
        )
    )


@dsl_user_op
def _s32_as_f32(value: Int32, *, loc=None, ip=None) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(), [value.ir_value()], "mov.b32 $0, $1;", "=f,r", asm_dialect=0, loc=loc, ip=ip
        )
    )


@dsl_user_op
def _st_release_sys_s32(address: Int64, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value()],
        "st.release.sys.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None) -> None:
    """Order this thread's earlier generic-proxy acquires before its later async-proxy (TMA) reads."""
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.global;",
        "",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


class _OneRoundRouter(KernelComponent):
    """One-round push dispatch with the ``_MetadataPushRouter`` output contract (see the module docstring)."""

    threads_per_cta: ClassVar[int] = 512
    warps_per_cta: ClassVar[int] = 16
    # Record layout, low word: token (16 bits) | top-k slot (6 bits) | local expert (10 bits); high word: score bits.
    record_token_bits: ClassVar[int] = 16
    record_slot_bits: ClassVar[int] = 6
    record_expert_bits: ClassVar[int] = 10
    max_world_size: ClassVar[int] = 64
    wait_sleep_cycles: ClassVar[int] = 64
    # Records per chunk the sorter keeps in registers between its two passes (the rest re-read with a scratch slot).
    sort_keep: ClassVar[int] = 8

    # Symmetric regions: sources write them through the peer pointer mapper.
    inbox_records_region: ClassVar[str] = "nvlink.one_round.inbox_records"
    inbox_count_region: ClassVar[str] = "nvlink.one_round.inbox_count"
    # Per-workspace call epoch; every rank advances it once per call, so it agrees across ranks.
    epoch_region: ClassVar[str] = "nvlink.one_round.epoch"
    # inbox_count word: epoch (low 12 bits) << 20 | records + 1. One producer per word and no reset: a word belongs to
    # call n exactly when its epoch field is n's. Two banks by epoch parity: without a kernel-wide barrier between
    # calls (data-driven drains), a peer with no rows on this rank can publish call n+1 before this rank's sorter read
    # its call-n word; it can never get two calls ahead (its call n+1 sort waits for this rank's call n+1 count).
    count_bits: ClassVar[int] = 20
    plane_activation_region: ClassVar[str] = "nvlink.one_round.plane_activation"
    plane_activation_sf_region: ClassVar[str] = "nvlink.one_round.plane_activation_sf"
    # Local regions: the router-to-main contract.
    sizes_region: ClassVar[str] = "nvlink.one_round.sizes"
    pool_expert_base_region: ClassVar[str] = "nvlink.one_round.pool_expert_base"
    token_src_metadata_region: ClassVar[str] = "nvlink.one_round.token_src_metadata"  # nosec B105
    fc1_topk_scores_region: ClassVar[str] = "nvlink.one_round.fc1_topk_scores"
    sort_slot_scratch_region: ClassVar[str] = "nvlink.one_round.sort_slot_scratch"
    sizes_ready_region: ClassVar[str] = "nvlink.one_round.sizes_ready"
    metadata_ready_region: ClassVar[str] = "nvlink.one_round.metadata_ready"
    # SMEM regions of the communication launch.
    smem_destination_count_region: ClassVar[str] = "nvlink.one_round.smem.destination_count"
    smem_histogram_region: ClassVar[str] = "nvlink.one_round.smem.histogram"
    smem_padded_region: ClassVar[str] = "nvlink.one_round.smem.padded"
    smem_prefix_region: ClassVar[str] = "nvlink.one_round.smem.prefix"
    smem_warp_totals_region: ClassVar[str] = "nvlink.one_round.smem.warp_totals"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "world_size": int,
            "expert_count": int,
            "topk": int,
            "max_tokens_per_rank": int,
            "hidden_size": int,
            "quant_kind": str,
            "apply_topk_at_fc1": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "token_padding_block": int,
            "one_round_cta_count": OptionalRequirement(int),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)
        self.world_size = problem_desc["world_size"]
        self.expert_count = problem_desc["expert_count"]
        self.topk = problem_desc["topk"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]
        self.token_padding_block = impl_desc["token_padding_block"]
        self.one_round_cta_count = int(impl_desc.get("one_round_cta_count", 2))
        self._validate_configuration()

        self.experts_per_rank = self.expert_count // self.world_size
        self.local_expert_count_padded = round_up(self.experts_per_rank, 4)
        self.world_size_padded = round_up(self.world_size, 4)
        self.chunk_count = self.world_size * self.one_round_cta_count
        self.global_warp_count = self.warps_per_cta * self.one_round_cta_count
        self.token_rounds = ceil_div(self.max_tokens_per_rank, self.global_warp_count)
        self.records_per_chunk = (
            self.token_rounds * self.warps_per_cta * min(self.topk, self.experts_per_rank)
        )
        self.bytes_per_token = self.hidden_size * int(self.quant_kind.activation_dtype.width) // 8
        valid_sf = self.hidden_size // self.quant_kind.sf_vec_size
        sf_elements_per_16_bytes = 128 // int(self.quant_kind.sf_dtype.width)
        self.sf_row_bytes = (
            round_up(valid_sf, sf_elements_per_16_bytes) * int(self.quant_kind.sf_dtype.width) // 8
        )
        self.activation_vectors = self.bytes_per_token // 16
        self.sf_vectors = self.sf_row_bytes // 16
        self.worst_case_token_count = self.worst_case_padded_tokens(self.token_padding_block)
        self._smem_workspace = self._build_smem_workspace()

    def _validate_configuration(self) -> None:
        if self.world_size > self.max_world_size:
            raise NotImplementedError(
                f"One-round dispatch supports at most {self.max_world_size} ranks."
            )
        if self.world_size * self.one_round_cta_count > self.threads_per_cta:
            raise NotImplementedError(
                "world_size * one_round_cta_count must fit one CTA's threads."
            )
        if self.max_tokens_per_rank > (1 << self.record_token_bits):
            raise NotImplementedError("One-round dispatch packs the token index into 16 bits.")
        if self.topk > 32:
            raise NotImplementedError("One-round dispatch reads one top-k slot per lane of a warp.")
        if self.expert_count // self.world_size > (1 << self.record_expert_bits):
            raise NotImplementedError("One-round dispatch packs the local expert into 10 bits.")
        if (self.hidden_size * int(self.quant_kind.activation_dtype.width) // 8) % 16 != 0:
            raise ValueError("One-round dispatch moves activation rows in 16-byte vectors.")

    def worst_case_padded_tokens(self, block: int) -> int:
        """Same pool bound as ``_MetadataPushRouter.worst_case_padded_tokens``."""
        source_token_capacity = self.world_size * self.max_tokens_per_rank
        experts_per_rank = self.expert_count // self.world_size
        routes_per_source_token = min(self.topk, experts_per_rank)
        route_capacity = source_token_capacity * routes_per_source_token
        active_expert_capacity = min(experts_per_rank, route_capacity)
        route_budget_blocks = (
            active_expert_capacity + (route_capacity - active_expert_capacity) // block
        )
        expert_bound_blocks = active_expert_capacity * int(ceil_div(source_token_capacity, block))
        return min(route_budget_blocks, expert_bound_blocks) * block

    @property
    def metadata_ready_target(self) -> int:
        return 1

    @property
    def router_smem_workspace(self) -> SmemWorkspace:
        return self._smem_workspace

    @property
    def router_grid_cta_count(self) -> int:
        return self.one_round_cta_count

    def _build_smem_workspace(self) -> SmemWorkspace:
        workspace = SmemWorkspace()
        workspace.register_tensor(
            self.smem_destination_count_region,
            cutlass.Int32,
            (self.world_size_padded,),
            byte_alignment=16,
        )
        for region in (
            self.smem_histogram_region,
            self.smem_padded_region,
            self.smem_prefix_region,
        ):
            workspace.register_tensor(
                region, cutlass.Int32, (self.local_expert_count_padded,), byte_alignment=16
            )
        workspace.register_tensor(
            self.smem_warp_totals_region, cutlass.Int32, (self.warps_per_cta,), byte_alignment=16
        )
        workspace.finalize(max_bytes=48 * 1024)
        return workspace

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        workspace.register(
            self.inbox_records_region,
            cutlass.Int64,
            (self.chunk_count * self.records_per_chunk,),
            buffer_space="shared",
            byte_alignment=16,
        )
        workspace.register(
            self.inbox_count_region,
            cutlass.Int32,
            (2 * self.chunk_count,),
            buffer_space="shared",
            reset="zero_on_first_allocate",
        )
        # In the symmetric buffer (written by this rank only) so it lives and dies with the inbox it tags.
        workspace.register(
            self.epoch_region,
            cutlass.Int32,
            (1,),
            buffer_space="shared",
            reset="zero_on_first_allocate",
        )
        workspace.register(
            self.plane_activation_region,
            cutlass.Int32,
            (self.world_size * self.max_tokens_per_rank * self.bytes_per_token // 4,),
            buffer_space="shared",
            byte_alignment=128,
        )
        workspace.register(
            self.plane_activation_sf_region,
            cutlass.Int32,
            (self.world_size * self.max_tokens_per_rank * self.sf_row_bytes // 4,),
            buffer_space="shared",
            byte_alignment=128,
        )
        workspace.register(
            self.sizes_region,
            cutlass.Int32,
            (self.local_expert_count_padded,),
            buffer_space="local",
        )
        workspace.register(
            self.sort_slot_scratch_region,
            cutlass.Int32,
            (self.chunk_count * self.records_per_chunk,),
            buffer_space="local",
        )
        workspace.register(
            self.pool_expert_base_region,
            cutlass.Int32,
            (self.local_expert_count_padded,),
            buffer_space="local",
        )
        workspace.register(
            self.token_src_metadata_region,
            cutlass.Int64,
            (self.worst_case_token_count,),
            buffer_space="local",
            byte_alignment=16,
        )
        if self.apply_topk_at_fc1:
            workspace.register(
                self.fc1_topk_scores_region,
                cutlass.Float32,
                (self.worst_case_token_count,),
                buffer_space="local",
            )
        workspace.register(
            self.sizes_ready_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
        )
        workspace.register(
            self.metadata_ready_region,
            cutlass.Int32,
            (1,),
            buffer_space="local",
            reset="tail_reset",
        )

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "_OneRoundRouter":
        if values:
            raise ValueError("_OneRoundRouter carries no MLIR values.")
        return self

    # ------------------------------------------------------------------
    # Main-kernel contract (same names as ``_MetadataPushRouter``)
    # ------------------------------------------------------------------

    @cute.jit
    def local_expert_sizes(
        self, device_workspace: DeviceWorkspace, local_rank: Int32
    ) -> cute.Tensor:
        sizes = device_workspace.tensor(self.sizes_region)
        return cute.make_tensor(sizes.iterator, cute.make_layout((self.experts_per_rank,)))

    @cute.jit
    def pool_expert_base_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.pool_expert_base_region)

    @cute.jit
    def token_src_metadata_pointer(self, device_workspace: DeviceWorkspace) -> cute.Pointer:
        return device_workspace.ptr(self.token_src_metadata_region)

    @cute.jit
    def token_src_metadata_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.token_src_metadata_region)

    @cute.jit
    def fc1_topk_scores_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.apply_topk_at_fc1):
            return None
        return device_workspace.tensor(self.fc1_topk_scores_region)

    @cute.jit
    def plane_activation_pointer(self, device_workspace: DeviceWorkspace) -> cute.Pointer:
        return device_workspace.ptr(self.plane_activation_region)

    @cute.jit
    def plane_activation_sf_pointer(self, device_workspace: DeviceWorkspace) -> cute.Pointer:
        return device_workspace.ptr(self.plane_activation_sf_region)

    @cute.jit
    def _wait_flag(self, pointer: cute.Pointer, sleep_cycles: int) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        if thread_idx % Int32(32) == Int32(0):
            while cute.arch.load(pointer, Int32, sem="acquire", scope="gpu") == Int32(0):
                nanosleep(sleep_cycles)
        cute.arch.sync_warp()

    @cute.jit
    def wait_for_sizes_ready(
        self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000
    ) -> None:
        del (
            sleep_cycles
        )  # The local flag lands within a few microseconds; poll tighter than the router does.
        self._wait_flag(device_workspace.ptr(self.sizes_ready_region), self.wait_sleep_cycles)

    @cute.jit
    def wait_for_metadata_ready(self, device_workspace: DeviceWorkspace) -> None:
        self._wait_flag(device_workspace.ptr(self.metadata_ready_region), self.wait_sleep_cycles)

    # ------------------------------------------------------------------
    # Communication launch
    # ------------------------------------------------------------------

    @cute.jit
    def launch_router(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper_host,
        device_workspace: DeviceWorkspace,
        stream: cuda.CUstream,
    ) -> None:
        """Launch the push + sort grid. Not PDL: it starts after its predecessor completed, so it may read the
        staged top-k and activation at entry; it releases the MoE kernel at entry like the metadata-push router."""
        kernel = self._kernel(
            topk_indices,
            topk_scores,
            activation,
            activation_sf,
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper_host.make_device_object(),
            device_workspace,
        )
        kernel.launch(
            grid=[self.one_round_cta_count, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def _kernel(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        device_workspace: DeviceWorkspace,
    ) -> None:
        cute.arch.griddepcontrol_launch_dependents()
        thread_idx, _, _ = cute.arch.thread_idx()
        cta_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        lane_idx = thread_idx % Int32(32)

        storage = cutlass.utils.SmemAllocator().allocate(self._smem_workspace.storage_class())
        smem_base = storage.buffer.data_ptr()
        device_workspace.assign_device_members(local_workspace, shared_workspace)

        self._push(
            topk_indices,
            topk_scores,
            activation,
            activation_sf,
            local_rank,
            peer_rank_ptr_mapper,
            device_workspace,
            smem_base,
            cta_idx,
            thread_idx,
            warp_idx,
            lane_idx,
        )
        if cta_idx == Int32(0):
            self._sort(device_workspace, smem_base, thread_idx, warp_idx, lane_idx)
        device_workspace.remove_device_members()

    @cute.jit
    def _push(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        activation: cute.Tensor,
        activation_sf: cute.Tensor,
        local_rank: Int32,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        cta_idx: Int32,
        thread_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
    ) -> None:
        destination_count = self._smem_workspace.tensor(
            self.smem_destination_count_region, smem_base
        )
        for zero_round in cutlass.range_constexpr(
            ceil_div(self.world_size_padded, self.threads_per_cta)
        ):
            index = Int32(zero_round * self.threads_per_cta) + thread_idx
            if index < Int32(self.world_size_padded):
                destination_count[index] = Int32(0)
        cute.arch.sync_threads()

        chunk = local_rank * Int32(self.one_round_cta_count) + cta_idx
        records_address = device_workspace.ptr(self.inbox_records_region).toint() + Int64(
            chunk
        ) * Int64(self.records_per_chunk * 8)
        plane_activation_address = device_workspace.ptr(self.plane_activation_region).toint()
        plane_sf_address = device_workspace.ptr(self.plane_activation_sf_region).toint()
        activation_address = activation.iterator.toint()
        # Tokens interleave over the CTAs first, so a few live tokens still spread their pushes over every SM of
        # the grid (one SM's remote-store rate, ~50-60 GB/s, is what bounds a push).
        global_warp_idx = warp_idx * Int32(self.one_round_cta_count) + cta_idx

        for token_round in cutlass.range_constexpr(self.token_rounds):
            token = global_warp_idx + Int32(token_round * self.global_warp_count)
            if token < Int32(self.max_tokens_per_rank):
                lane_valid = cutlass.Boolean(False)
                expert = Int32(0)
                score_bits = Int32(0)
                if lane_idx < Int32(self.topk):
                    raw_index = topk_indices[token, lane_idx]
                    # Int64 ids; a negative one compares as a large unsigned value.
                    lane_valid = cutlass.Uint64(raw_index) < cutlass.Uint64(self.expert_count)
                    expert = Int32(raw_index)
                    if cutlass.const_expr(self.apply_topk_at_fc1):
                        score_bits = _f32_as_s32(cutlass.Float32(topk_scores[token, lane_idx]))
                destination = expert // Int32(self.experts_per_rank)
                local_expert = expert - destination * Int32(self.experts_per_rank)
                low_bit = Int32(0)
                high_bit = Int32(0)
                if lane_valid:
                    if destination < Int32(32):
                        low_bit = Int32(1) << destination
                    else:
                        high_bit = Int32(1) << (destination - Int32(32))
                low_mask = Int32(cute.arch.warp_redux_sync(low_bit, "or"))
                high_mask = Int32(0)
                if cutlass.const_expr(self.world_size > 32):
                    high_mask = Int32(cute.arch.warp_redux_sync(high_bit, "or"))

                if lane_valid:
                    position = Int32(
                        cute.arch.atomic_add(
                            destination_count.iterator + destination,
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                    )
                    low_word = (
                        token
                        | (lane_idx << Int32(self.record_token_bits))
                        | (local_expert << Int32(self.record_token_bits + self.record_slot_bits))
                    )
                    record = (Int64(score_bits) << Int64(32)) | (
                        Int64(low_word) & Int64(0xFFFFFFFF)
                    )
                    stg_b64(
                        peer_rank_ptr_mapper.map(
                            records_address + Int64(position) * Int64(8), destination, Int64(0)
                        ),
                        record,
                    )

                if (low_mask | high_mask) != Int32(0):
                    self._push_row(
                        activation_address + Int64(token) * Int64(self.bytes_per_token),
                        activation_sf[Int64(token), None].iterator.toint(),
                        plane_activation_address
                        + (Int64(local_rank) * Int64(self.max_tokens_per_rank) + Int64(token))
                        * Int64(self.bytes_per_token),
                        plane_sf_address
                        + (Int64(local_rank) * Int64(self.max_tokens_per_rank) + Int64(token))
                        * Int64(self.sf_row_bytes),
                        low_mask,
                        high_mask,
                        peer_rank_ptr_mapper,
                        lane_idx,
                    )

        # Every payload and record of this CTA is issued. The CTA barrier carries the other threads' stores into the
        # publishing warp, whose release store (one MEMBAR.SYS per warp) orders them before each count word; a
        # separate fence in thread 0 would add a second system fence on the same warp.
        cute.arch.sync_warp()
        cute.arch.sync_threads()
        epoch = cute.arch.load(device_workspace.ptr(self.epoch_region), Int32)
        epoch_tag = (epoch & Int32(0xFFF)) << Int32(self.count_bits)
        count_address = device_workspace.ptr(self.inbox_count_region).toint() + (
            Int64(epoch & Int32(1)) * Int64(self.chunk_count) + Int64(chunk)
        ) * Int64(4)
        for publish_round in cutlass.range_constexpr(
            ceil_div(self.world_size, self.threads_per_cta)
        ):
            destination = Int32(publish_round * self.threads_per_cta) + thread_idx
            if destination < Int32(self.world_size):
                _st_release_sys_s32(
                    peer_rank_ptr_mapper.map(count_address, destination, Int64(0)),
                    epoch_tag | (destination_count[destination] + Int32(1)),
                )

    @cute.jit
    def _push_row(
        self,
        source_activation: Int64,
        source_sf: Int64,
        plane_activation_row: Int64,
        plane_sf_row: Int64,
        low_mask: Int32,
        high_mask: Int32,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        lane_idx: Int32,
    ) -> None:
        """Load one token's activation and scale rows into registers, then store them once into every
        destination rank of the warp-uniform masks."""
        activation_rounds = ceil_div(self.activation_vectors, 32)
        sf_rounds = ceil_div(self.sf_vectors, 32)
        values = cute.make_rmem_tensor(((activation_rounds + sf_rounds) * 4,), cutlass.Int32)
        for vector_round in cutlass.range_constexpr(activation_rounds):
            vector = Int32(vector_round * 32) + lane_idx
            if vector < Int32(self.activation_vectors):
                v = _ldg_v4_b32(source_activation + Int64(vector) * Int64(16))
                for element in cutlass.range_constexpr(4):
                    values[vector_round * 4 + element] = v[element]
        for vector_round in cutlass.range_constexpr(sf_rounds):
            vector = Int32(vector_round * 32) + lane_idx
            if vector < Int32(self.sf_vectors):
                v = _ldg_v4_b32(source_sf + Int64(vector) * Int64(16))
                for element in cutlass.range_constexpr(4):
                    values[(activation_rounds + vector_round) * 4 + element] = v[element]

        for mask_word in cutlass.range_constexpr(2 if self.world_size > 32 else 1):
            remaining = low_mask if mask_word == 0 else high_mask
            while remaining != Int32(0):
                lowest = remaining & (Int32(0) - remaining)
                destination = Int32(cute.arch.popc(lowest - Int32(1))) + Int32(mask_word * 32)
                remaining = remaining ^ lowest
                peer_offset = peer_rank_ptr_mapper.map(Int64(0), destination, Int64(0))
                for vector_round in cutlass.range_constexpr(activation_rounds):
                    vector = Int32(vector_round * 32) + lane_idx
                    if vector < Int32(self.activation_vectors):
                        _stg_v4_b32(
                            plane_activation_row + peer_offset + Int64(vector) * Int64(16),
                            values[vector_round * 4 + 0],
                            values[vector_round * 4 + 1],
                            values[vector_round * 4 + 2],
                            values[vector_round * 4 + 3],
                        )
                for vector_round in cutlass.range_constexpr(sf_rounds):
                    vector = Int32(vector_round * 32) + lane_idx
                    if vector < Int32(self.sf_vectors):
                        base = (activation_rounds + vector_round) * 4
                        _stg_v4_b32(
                            plane_sf_row + peer_offset + Int64(vector) * Int64(16),
                            values[base + 0],
                            values[base + 1],
                            values[base + 2],
                            values[base + 3],
                        )

    @cute.jit
    def _sort(
        self,
        device_workspace: DeviceWorkspace,
        smem_base: cute.Pointer,
        thread_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
    ) -> None:
        """One thread per (source rank, source CTA) chunk: it waits for that chunk's count word, loads the chunk's
        records as soon as they are there and takes each record's slot inside its expert from the histogram atomic
        (no second counting pass); after the expert prefix it writes the pool rows from the records it kept in
        registers (sort_keep of them; the rest re-read with their slot from a scratch row)."""
        histogram = self._smem_workspace.tensor(self.smem_histogram_region, smem_base)
        padded = self._smem_workspace.tensor(self.smem_padded_region, smem_base)
        prefix = self._smem_workspace.tensor(self.smem_prefix_region, smem_base)
        warp_totals = self._smem_workspace.tensor(self.smem_warp_totals_region, smem_base)

        for zero_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_count_padded, self.threads_per_cta)
        ):
            expert = Int32(zero_round * self.threads_per_cta) + thread_idx
            if expert < Int32(self.local_expert_count_padded):
                histogram[expert] = Int32(0)
        cute.arch.sync_threads()

        epoch = cute.arch.load(device_workspace.ptr(self.epoch_region), Int32)
        count_address = device_workspace.ptr(self.inbox_count_region).toint() + Int64(
            epoch & Int32(1)
        ) * Int64(self.chunk_count * 4)
        epoch_tag = (epoch & Int32(0xFFF)) << Int32(self.count_bits)
        count_mask = Int32((1 << self.count_bits) - 1)
        epoch_mask = Int32(-(1 << self.count_bits))
        records_address = device_workspace.ptr(self.inbox_records_region).toint()
        scratch_address = device_workspace.ptr(self.sort_slot_scratch_region).toint()
        expert_shift = Int32(self.record_token_bits + self.record_slot_bits)
        expert_mask = Int32((1 << self.record_expert_bits) - 1)
        keep = self.sort_keep
        kept_records = cute.make_rmem_tensor((keep,), cutlass.Int64)
        kept_slots = cute.make_rmem_tensor((keep,), cutlass.Int32)
        record_count = Int32(0)
        chunk_address = records_address + Int64(thread_idx) * Int64(self.records_per_chunk * 8)
        chunk_scratch = scratch_address + Int64(thread_idx) * Int64(self.records_per_chunk * 4)

        # Pass 1: counts, record loads and in-expert slots, chunk by chunk as they arrive.
        if thread_idx < Int32(self.chunk_count):
            published = _ld_acquire_sys_s32(count_address + Int64(thread_idx) * Int64(4))
            while ((published & epoch_mask) != epoch_tag) | ((published & count_mask) == Int32(0)):
                nanosleep(self.wait_sleep_cycles)
                published = _ld_acquire_sys_s32(count_address + Int64(thread_idx) * Int64(4))
            record_count = (published & count_mask) - Int32(1)
            for slot in cutlass.range_constexpr(keep):
                if Int32(slot) < record_count:
                    kept_records[slot] = _ld_b64(chunk_address + Int64(slot * 8))
            for slot in cutlass.range_constexpr(keep):
                if Int32(slot) < record_count:
                    low = Int32(kept_records[slot] & Int64(0xFFFFFFFF))
                    kept_slots[slot] = Int32(
                        cute.arch.atomic_add(
                            histogram.iterator + ((low >> expert_shift) & expert_mask),
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
                    )
            for extra in cutlass.range(record_count - Int32(keep), unroll=1):
                record = _ld_b64(chunk_address + Int64(Int32(keep) + extra) * Int64(8))
                low = Int32(record & Int64(0xFFFFFFFF))
                in_expert = Int32(
                    cute.arch.atomic_add(
                        histogram.iterator + ((low >> expert_shift) & expert_mask),
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                )
                cute.arch.store(
                    cute.make_ptr(
                        Int32,
                        chunk_scratch + Int64(Int32(keep) + extra) * Int64(4),
                        AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    in_expert,
                )
        cute.arch.sync_warp()
        cute.arch.sync_threads()

        for expert_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_count_padded, self.threads_per_cta)
        ):
            expert = Int32(expert_round * self.threads_per_cta) + thread_idx
            if expert < Int32(self.local_expert_count_padded):
                padded[expert] = (
                    (histogram[expert] + Int32(self.token_padding_block - 1))
                    // Int32(self.token_padding_block)
                ) * Int32(self.token_padding_block)
        cute.arch.sync_threads()
        smem_exclusive_prefix(
            padded, prefix, warp_totals, self.threads_per_cta, thread_idx, lane_idx, warp_idx
        )

        sizes = device_workspace.tensor(self.sizes_region)
        pool_expert_base = device_workspace.tensor(self.pool_expert_base_region)
        for expert_round in cutlass.range_constexpr(
            ceil_div(self.local_expert_count_padded, self.threads_per_cta)
        ):
            expert = Int32(expert_round * self.threads_per_cta) + thread_idx
            if expert < Int32(self.local_expert_count_padded):
                sizes[expert] = histogram[expert]
                pool_expert_base[expert] = prefix[expert]
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            cute.arch.atomic_add(
                device_workspace.ptr(self.sizes_ready_region), Int32(1), sem="release", scope="gpu"
            )

        # Pass 2: pool rows of this thread's chunk.
        metadata_address = device_workspace.ptr(self.token_src_metadata_region).toint()
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_address = device_workspace.ptr(self.fc1_topk_scores_region).toint()
        if thread_idx < Int32(self.chunk_count):
            source_rank = thread_idx // Int32(self.one_round_cta_count)
            for slot in cutlass.range_constexpr(keep):
                if Int32(slot) < record_count:
                    self._write_pool_row(
                        kept_records[slot],
                        kept_slots[slot],
                        source_rank,
                        prefix,
                        metadata_address,
                        score_address if cutlass.const_expr(self.apply_topk_at_fc1) else None,
                    )
            for extra in cutlass.range(record_count - Int32(keep), unroll=1):
                self._write_pool_row(
                    _ld_b64(chunk_address + Int64(Int32(keep) + extra) * Int64(8)),
                    cute.arch.load(
                        cute.make_ptr(
                            Int32,
                            chunk_scratch + Int64(Int32(keep) + extra) * Int64(4),
                            AddressSpace.gmem,
                            assumed_align=4,
                        ),
                        Int32,
                    ),
                    source_rank,
                    prefix,
                    metadata_address,
                    score_address if cutlass.const_expr(self.apply_topk_at_fc1) else None,
                )
        cute.arch.sync_warp()
        cute.arch.sync_threads()
        if thread_idx == Int32(0):
            cute.arch.atomic_add(
                device_workspace.ptr(self.metadata_ready_region),
                Int32(1),
                sem="release",
                scope="gpu",
            )
            # Every CTA of this grid read the epoch before it published (the sorter counted all of them).
            cute.arch.store(device_workspace.ptr(self.epoch_region), epoch + Int32(1))

    @cute.jit
    def _write_pool_row(
        self,
        record: Int64,
        in_expert: Int32,
        source_rank: Int32,
        prefix: cute.Tensor,
        metadata_address: Int64,
        score_address: Optional[Int64],
    ) -> None:
        low = Int32(record & Int64(0xFFFFFFFF))
        local_expert = (low >> Int32(self.record_token_bits + self.record_slot_bits)) & Int32(
            (1 << self.record_expert_bits) - 1
        )
        pool_row = prefix[local_expert] + in_expert
        metadata = TokenSrcMetadata(
            src_rank=source_rank,
            src_token=low & Int32((1 << self.record_token_bits) - 1),
            src_topk=(low >> Int32(self.record_token_bits))
            & Int32((1 << self.record_slot_bits) - 1),
        )
        stg_b64(
            metadata_address + Int64(pool_row) * Int64(TokenSrcMetadata.nbytes), metadata.pack()
        )
        if cutlass.const_expr(score_address is not None):
            stg_f32(
                score_address + Int64(pool_row) * Int64(4), _s32_as_f32(Int32(record >> Int64(32)))
            )


__all__ = ["_OneRoundRouter", "fence_proxy_async_global"]
