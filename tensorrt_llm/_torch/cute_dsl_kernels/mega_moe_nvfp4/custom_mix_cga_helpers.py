# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed-CGA TMA helpers and TMA-to-UMMA pipeline."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.cpasync.copy import (
    TMA_DESC_PTR_FIELD_NAME,
    TMA_MBAR_PTR_FIELD_NAME,
    TMA_MCAST_MASK_FIELD_NAME,
    CopyBulkTensorTileG2SMulticastNonExecTrait,
    CopyBulkTensorTileG2SMulticastTrait,
    CopyBulkTensorTileG2SNonExecTrait,
    CopyBulkTensorTileG2STrait,
)
from cutlass.cutlass_dsl import Boolean, Int16, Int32, dsl_user_op, if_generate

TmaAtomOrPair = Union[cute.CopyAtom, Tuple[cute.CopyAtom, cute.CopyAtom]]


class _RuntimeInitMbarrierArray(pipeline.MbarrierArray):
    """Initialize an mbarrier array once with a runtime arrival count."""

    @dsl_user_op
    def __init__(
        self,
        barrier_storage: cute.Pointer,
        num_stages: int,
        agent: Tuple[pipeline.PipelineOp, pipeline.CooperativeGroup],
        init_arrive_count: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        self.barrier_storage = barrier_storage
        self.tx_count = 0
        self.num_stages = num_stages
        self.op_type, self.cg = agent
        # CUTLASS DSL 4.5 requires static participant metadata. The hardware
        # barrier itself is initialized with init_arrive_count below.
        self.arrive_count = self.cg.size
        mbarrier_layout = getattr(pipeline, "MbarrierLayout", None)
        self.mbarrier_layout = mbarrier_layout.V0 if mbarrier_layout is not None else None
        self.name = ""
        self.mbarrier_base = barrier_storage

        def initialize_barriers() -> None:
            with cute.arch.elect_one(loc=loc, ip=ip):
                for stage in range(self.num_stages):
                    cute.arch.mbarrier_init(
                        self.get_barrier(stage, loc=loc, ip=ip),
                        init_arrive_count,
                        loc=loc,
                        ip=ip,
                    )

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx(loc=loc, ip=ip), loc=loc, ip=ip)
        if_generate(warp_idx == 0, initialize_barriers, loc=loc, ip=ip)

    def __new_from_mlir_values__(self, values: list[object]) -> "_RuntimeInitMbarrierArray":
        if len(values) != 1:
            raise ValueError(f"Expected one barrier storage value, got {len(values)}.")
        result = object.__new__(_RuntimeInitMbarrierArray)
        result.barrier_storage = values[0]
        result.tx_count = self.tx_count
        result.num_stages = self.num_stages
        result.op_type = self.op_type
        result.cg = self.cg
        result.arrive_count = self.arrive_count
        result.mbarrier_layout = self.mbarrier_layout
        result.name = self.name
        result.mbarrier_base = values[0]
        return result


@cute.jit
def make_runtime_init_pipeline_async(
    *,
    barrier_storage: cute.Pointer,
    num_stages: int,
    static_consumer_group_size: int,
    consumer_arrive_count: Int32,
    defer_sync: bool = False,
) -> pipeline.PipelineAsync:
    """Create an async pipeline whose consumer arrival count is runtime-selected."""
    if cutlass.const_expr(not isinstance(barrier_storage, cute.Pointer)):
        raise TypeError(f"barrier_storage must be a cute.Pointer, got {type(barrier_storage)}.")
    if cutlass.const_expr(
        isinstance(static_consumer_group_size, bool)
        or not isinstance(static_consumer_group_size, int)
        or static_consumer_group_size <= 0
    ):
        raise ValueError("static_consumer_group_size must be a positive Python int.")

    producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
    consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, static_consumer_group_size)
    full_sync_object = pipeline.MbarrierArray(
        barrier_storage=barrier_storage.align(min_align=8),
        num_stages=num_stages,
        agent=(pipeline.PipelineOp.AsyncThread, producer_group),
    )
    empty_sync_object = _RuntimeInitMbarrierArray(
        barrier_storage=barrier_storage.align(min_align=8) + num_stages,
        num_stages=num_stages,
        agent=(pipeline.PipelineOp.AsyncThread, consumer_group),
        init_arrive_count=consumer_arrive_count,
    )

    if cutlass.const_expr(not defer_sync):
        cute.arch.mbarrier_init_fence()
        pipeline.agent_sync(pipeline.Agent.ThreadBlock)

    return pipeline.PipelineAsync(
        sync_object_full=full_sync_object,
        sync_object_empty=empty_sync_object,
        num_stages=num_stages,
        producer_mask=None,
        consumer_mask=None,
    )


@cute.jit
def make_executable_tma_atom(atom: cute.CopyAtom) -> cute.CopyAtom:
    """Materialize a bare executable TMA atom without per-issue fields."""
    executable_value = _cute_nvgpu_ir.atom_make_exec_tma(atom._trait.value)
    if cutlass.const_expr(isinstance(atom._trait, CopyBulkTensorTileG2SMulticastNonExecTrait)):
        executable_trait = CopyBulkTensorTileG2SMulticastTrait(executable_value)
    elif cutlass.const_expr(isinstance(atom._trait, CopyBulkTensorTileG2SNonExecTrait)):
        executable_trait = CopyBulkTensorTileG2STrait(executable_value)
    else:
        raise TypeError(f"Unsupported non-executable TMA load trait {type(atom._trait)}.")
    return cute.CopyAtom(atom.op, executable_trait)


@cute.jit
def bind_executable_tma_load_fields(
    atom: cute.CopyAtom,
    *,
    tma_bar_ptr: cute.Pointer,
    tma_desc_ptr: Optional[cute.Pointer],
    mcast_mask: Optional[Int32],
) -> cute.CopyAtom:
    """Bind per-issue fields to an already executable TMA load atom."""
    executable_value = atom._trait.value
    if cutlass.const_expr(mcast_mask is not None):
        mcast_attr = ir.Attribute.parse(
            f"#cute_nvgpu.atom_copy_field_tmaload<{TMA_MCAST_MASK_FIELD_NAME}>"
        )
        executable_value = _cute_nvgpu_ir.atom_set_value(
            executable_value, mcast_attr, Int16(mcast_mask).ir_value()
        )
    if cutlass.const_expr(tma_desc_ptr is not None):
        descriptor_attr = ir.Attribute.parse(
            f"#cute_nvgpu.atom_copy_field_tmaload<{TMA_DESC_PTR_FIELD_NAME}>"
        )
        executable_value = _cute_nvgpu_ir.atom_set_value(
            executable_value, descriptor_attr, tma_desc_ptr.value
        )
    barrier_attr = ir.Attribute.parse(
        f"#cute_nvgpu.atom_copy_field_tmaload<{TMA_MBAR_PTR_FIELD_NAME}>"
    )
    executable_value = _cute_nvgpu_ir.atom_set_value(
        executable_value, barrier_attr, tma_bar_ptr.value
    )
    return cute.CopyAtom(atom.op, atom._trait.__class__(executable_value))


def _multicast_group_count(
    shape_vmnk: Tuple[int, int, int, int], mcast_mode_mn: Tuple[int, int]
) -> int:
    mcast_a_count = shape_vmnk[2] if mcast_mode_mn[0] else 0
    mcast_b_count = shape_vmnk[1] if mcast_mode_mn[1] else 0
    overlap = 1 if mcast_mode_mn[0] and mcast_mode_mn[1] else 0
    group_count = mcast_a_count + mcast_b_count - overlap
    if group_count <= 0:
        raise ValueError("At least one TMA multicast dependency mode must be enabled.")
    return group_count


@cute.jit
def _reader_mask(
    cta_layout_vmnk: cute.Layout,
    cta_coord_vmnk: cute.Coord,
    mcast_mode_mn: Tuple[int, int],
    mma_cta_count: int,
) -> Int32:
    """Build the general reader union for non-single-sided cluster layouts.

    The single-sided MegaMoE path bypasses this helper and uses a full-cluster mask.
    """
    mask = Int32(0)
    if cutlass.const_expr(mcast_mode_mn[0]):
        mask = mask | Int32(
            cpasync.create_tma_multicast_mask(cta_layout_vmnk, cta_coord_vmnk, mcast_mode=2)
        )
    if cutlass.const_expr(mcast_mode_mn[1]):
        mask = mask | Int32(
            cpasync.create_tma_multicast_mask(cta_layout_vmnk, cta_coord_vmnk, mcast_mode=1)
        )
    if cutlass.const_expr(mma_cta_count == 2):
        peer_coord_vmnk = (cta_coord_vmnk[0] ^ Int32(1), *cta_coord_vmnk[1:])
        if cutlass.const_expr(mcast_mode_mn[0]):
            mask = mask | Int32(
                cpasync.create_tma_multicast_mask(cta_layout_vmnk, peer_coord_vmnk, mcast_mode=2)
            )
        if cutlass.const_expr(mcast_mode_mn[1]):
            mask = mask | Int32(
                cpasync.create_tma_multicast_mask(cta_layout_vmnk, peer_coord_vmnk, mcast_mode=1)
            )
    return mask


@cute.jit
def _active_cluster_uses_cluster_sync(
    preferred_cluster_shape_mn: Tuple[int, int],
    fallback_cluster_shape_mn: Tuple[int, int],
    is_fallback_cluster: Boolean,
) -> Boolean:
    preferred_uses_cluster = preferred_cluster_shape_mn[0] * preferred_cluster_shape_mn[1] > 1
    fallback_uses_cluster = fallback_cluster_shape_mn[0] * fallback_cluster_shape_mn[1] > 1
    cluster_sync_behavior_differs = preferred_uses_cluster != fallback_uses_cluster
    return Boolean(preferred_uses_cluster) ^ (
        is_fallback_cluster & Boolean(cluster_sync_behavior_differs)
    )


@cute.jit
def pipeline_init_arrive_mixed_cga(
    preferred_cluster_shape_mn: Tuple[int, int],
    fallback_cluster_shape_mn: Tuple[int, int],
    is_fallback_cluster: Boolean,
    is_relaxed: bool = False,
) -> None:
    """Fence mbarrier initialization and arrive on the active runtime cluster."""
    active_uses_cluster = _active_cluster_uses_cluster_sync(
        preferred_cluster_shape_mn, fallback_cluster_shape_mn, is_fallback_cluster
    )
    cute.arch.mbarrier_init_fence()

    if active_uses_cluster:
        if cutlass.const_expr(is_relaxed):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.cluster_arrive()


@cute.jit
def pipeline_init_wait_mixed_cga(
    preferred_cluster_shape_mn: Tuple[int, int],
    fallback_cluster_shape_mn: Tuple[int, int],
    is_fallback_cluster: Boolean,
) -> None:
    """Wait on the active runtime cluster or CTA."""
    active_uses_cluster = _active_cluster_uses_cluster_sync(
        preferred_cluster_shape_mn, fallback_cluster_shape_mn, is_fallback_cluster
    )
    if active_uses_cluster:
        cute.arch.cluster_wait()
    else:
        cute.arch.sync_threads()


def _tma_multicast_pattern(
    cluster_mn: Tuple[int, int],
    mma_cta_count: int,
    tensor_role: Literal["a", "b", "sfa", "sfb"],
) -> int:
    cluster_m, cluster_n = cluster_mn
    if tensor_role in ("a", "sfa"):
        return sum(1 << (cluster_n_idx * cluster_m) for cluster_n_idx in range(cluster_n))
    if tensor_role == "b":
        return sum(
            1 << (cluster_m_idx * mma_cta_count)
            for cluster_m_idx in range(cluster_m // mma_cta_count)
        )
    if tensor_role == "sfb":
        return (1 << cluster_m) - 1
    raise ValueError(f"Unsupported TMA tensor role {tensor_role!r}.")


@cute.jit
def tma_multicast_mask(
    preferred_cluster_mn: Tuple[int, int],
    fallback_cluster_mn: Optional[Tuple[int, int]],
    cta_coord_in_cluster: cute.Coord,
    is_preferred: Optional[Boolean],
    is_2cta: bool,
    tensor_role: Literal["a", "b", "sfa", "sfb"],
) -> Int16:
    """Build a preferred/fallback TMA multicast mask."""
    preferred_m, preferred_n = preferred_cluster_mn
    if cutlass.const_expr(preferred_m <= 0 or preferred_n <= 0 or preferred_m * preferred_n > 16):
        raise ValueError(f"Invalid preferred cluster shape {preferred_cluster_mn}.")

    mma_cta_count = 2 if cutlass.const_expr(is_2cta) else 1
    if cutlass.const_expr(preferred_m % mma_cta_count != 0):
        raise ValueError("Preferred cluster M must be divisible by the MMA CTA count.")

    preferred_pattern = _tma_multicast_pattern(preferred_cluster_mn, mma_cta_count, tensor_role)
    cta_m = Int32(cta_coord_in_cluster[0])
    active_pattern = Int32(preferred_pattern)
    active_cluster_m = Int32(preferred_m)
    if cutlass.const_expr(fallback_cluster_mn is not None):
        fallback_m, fallback_n = fallback_cluster_mn
        if cutlass.const_expr(fallback_m <= 0 or fallback_n <= 0 or fallback_m * fallback_n > 16):
            raise ValueError(f"Invalid fallback cluster shape {fallback_cluster_mn}.")
        if cutlass.const_expr(preferred_m % fallback_m != 0 or preferred_n % fallback_n != 0):
            raise ValueError(
                "Preferred cluster dimensions must be divisible by fallback dimensions."
            )
        if cutlass.const_expr(fallback_m % mma_cta_count != 0):
            raise ValueError("Fallback cluster M must be divisible by the MMA CTA count.")

        fallback_pattern = _tma_multicast_pattern(fallback_cluster_mn, mma_cta_count, tensor_role)
        if cutlass.const_expr(preferred_pattern != fallback_pattern):
            if cutlass.const_expr(is_preferred is None):
                raise ValueError(
                    "is_preferred is required when preferred and fallback "
                    "multicast patterns differ."
                )
            if cutlass.const_expr(
                tensor_role in ("b", "sfb")
                and fallback_pattern & preferred_pattern != fallback_pattern
            ):
                raise ValueError(
                    "Fallback B/SFB multicast pattern must be a subset of the preferred pattern."
                )
            if not is_preferred:
                active_pattern = Int32(fallback_pattern)
                active_cluster_m = Int32(fallback_m)

    if cutlass.const_expr(tensor_role in ("a", "sfa")):
        return Int16(active_pattern << cta_m)

    cta_n = Int32(cta_coord_in_cluster[1])
    mma_cta_index = (
        Int32(0) if cutlass.const_expr(tensor_role == "sfb" or mma_cta_count == 1) else cta_m % 2
    )
    offset = cta_n * active_cluster_m + mma_cta_index
    return Int16(active_pattern << offset)


@dataclass(frozen=True)
class PipelineTmaUmmaMixedCga(pipeline.PipelineAsync):
    """TMA-to-UMMA pipeline selected from preferred/fallback cluster layouts."""

    # One leader per UMMA V group, not one leader for the entire physical cluster.
    is_mma_leader_cta: Boolean
    cta_group: tcgen05.CtaGroup

    @staticmethod
    @cute.jit
    def create(
        *,
        barrier_storage: cute.Pointer,
        num_stages: int,
        producer_group: pipeline.CooperativeGroup,
        num_mma_consumer_warps: int,
        tx_count: int,
        preferred_cta_layout_vmnk: cute.Layout,
        fallback_cta_layout_vmnk: Optional[cute.Layout],
        preferred_shape_vmnk: Tuple[int, int, int, int],
        fallback_shape_vmnk: Tuple[int, int, int, int],
        cta_coord_vmnk: cute.Coord,
        is_fallback_cluster: Optional[Boolean],
        mcast_mode_mn: Tuple[int, int] = (1, 1),
        defer_sync: bool = False,
    ) -> "PipelineTmaUmmaMixedCga":
        if cutlass.const_expr(not isinstance(barrier_storage, cute.Pointer)):
            raise TypeError(f"barrier_storage must be a cute.Pointer, got {type(barrier_storage)}.")
        if cutlass.const_expr(num_stages <= 0):
            raise ValueError("num_stages must be positive.")
        if cutlass.const_expr(tx_count < 0):
            raise ValueError("tx_count must be nonnegative.")
        if cutlass.const_expr(
            isinstance(num_mma_consumer_warps, bool)
            or not isinstance(num_mma_consumer_warps, int)
            or num_mma_consumer_warps <= 0
        ):
            raise ValueError("num_mma_consumer_warps must be a positive Python int.")
        if cutlass.const_expr(
            len(mcast_mode_mn) != 2 or any(mode not in (0, 1) for mode in mcast_mode_mn)
        ):
            raise ValueError("mcast_mode_mn must contain two zero-or-one values.")

        fallback_layout = (
            preferred_cta_layout_vmnk
            if fallback_cta_layout_vmnk is None
            else fallback_cta_layout_vmnk
        )
        if cutlass.const_expr(
            not all(isinstance(extent, int) for extent in preferred_shape_vmnk)
            or not all(isinstance(extent, int) for extent in fallback_shape_vmnk)
        ):
            raise ValueError("Mixed-CGA pipeline layouts must have static VMNK shapes.")
        if cutlass.const_expr(preferred_shape_vmnk[0] != fallback_shape_vmnk[0]):
            raise ValueError("Preferred and fallback layouts must use the same MMA V extent.")
        if cutlass.const_expr(preferred_shape_vmnk[3] != 1 or fallback_shape_vmnk[3] != 1):
            raise ValueError("Mixed-CGA TMA-to-UMMA pipelines require cluster K=1.")

        mma_cta_count = preferred_shape_vmnk[0]
        if cutlass.const_expr(mma_cta_count not in (1, 2)):
            raise ValueError("The MMA V extent must be one or two.")
        cta_group = tcgen05.CtaGroup.TWO if mma_cta_count == 2 else tcgen05.CtaGroup.ONE
        is_true_mixed = preferred_shape_vmnk != fallback_shape_vmnk
        if cutlass.const_expr(is_true_mixed and is_fallback_cluster is None):
            raise ValueError(
                "is_fallback_cluster is required when preferred and fallback layouts differ."
            )

        preferred_group_count = _multicast_group_count(preferred_shape_vmnk, mcast_mode_mn)
        fallback_group_count = _multicast_group_count(fallback_shape_vmnk, mcast_mode_mn)
        preferred_arrival_count = preferred_group_count * num_mma_consumer_warps
        fallback_arrival_count = fallback_group_count * num_mma_consumer_warps

        preferred_cluster_size = (
            preferred_shape_vmnk[0] * preferred_shape_vmnk[1] * preferred_shape_vmnk[2]
        )
        fallback_cluster_size = (
            fallback_shape_vmnk[0] * fallback_shape_vmnk[1] * fallback_shape_vmnk[2]
        )
        tracks_both_operand_families = mcast_mode_mn == (1, 1)
        is_single_sided = tracks_both_operand_families and (
            (preferred_shape_vmnk[1] == fallback_shape_vmnk[1] == 1)
            or (preferred_shape_vmnk[2] == fallback_shape_vmnk[2] == 1)
        )

        active_arrival_count = Int32(preferred_arrival_count)
        consumer_mask = Int32((1 << preferred_cluster_size) - 1)
        if cutlass.const_expr(is_true_mixed):
            if is_fallback_cluster:
                active_arrival_count = Int32(fallback_arrival_count)
                if cutlass.const_expr(is_single_sided):
                    consumer_mask = Int32((1 << fallback_cluster_size) - 1)
                else:
                    consumer_mask = _reader_mask(
                        fallback_layout, cta_coord_vmnk, mcast_mode_mn, mma_cta_count
                    )
            else:
                active_arrival_count = Int32(preferred_arrival_count)
                if cutlass.const_expr(is_single_sided):
                    consumer_mask = Int32((1 << preferred_cluster_size) - 1)
                else:
                    consumer_mask = _reader_mask(
                        preferred_cta_layout_vmnk, cta_coord_vmnk, mcast_mode_mn, mma_cta_count
                    )
        elif cutlass.const_expr(not is_single_sided):
            consumer_mask = _reader_mask(
                preferred_cta_layout_vmnk, cta_coord_vmnk, mcast_mode_mn, mma_cta_count
            )

        full_sync_object = pipeline.MbarrierArray(
            barrier_storage=barrier_storage.align(min_align=8),
            num_stages=num_stages,
            agent=(pipeline.PipelineOp.TmaLoad, producer_group),
            tx_count=tx_count,
        )
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, preferred_arrival_count)
        empty_sync_object = _RuntimeInitMbarrierArray(
            barrier_storage=barrier_storage.align(min_align=8) + num_stages,
            num_stages=num_stages,
            agent=(pipeline.PipelineOp.TCGen05Mma, consumer_group),
            init_arrive_count=active_arrival_count,
        )

        if cutlass.const_expr(not defer_sync):
            cute.arch.mbarrier_init_fence()
            active_cluster_size = Int32(preferred_cluster_size)
            if cutlass.const_expr(is_true_mixed):
                if is_fallback_cluster:
                    active_cluster_size = Int32(fallback_cluster_size)
            if active_cluster_size == Int32(1):
                pipeline.agent_sync(pipeline.Agent.ThreadBlock)
            else:
                pipeline.agent_sync(pipeline.Agent.ThreadBlockCluster, is_relaxed=True)

        return PipelineTmaUmmaMixedCga(
            sync_object_full=full_sync_object,
            sync_object_empty=empty_sync_object,
            num_stages=num_stages,
            producer_mask=None,
            consumer_mask=consumer_mask,
            is_mma_leader_cta=cta_coord_vmnk[0] == Int32(0),
            cta_group=cta_group,
        )

    @cute.jit
    def producer_acquire(
        self,
        state: pipeline.PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        *,
        expected_tx: Optional[Int32] = None,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        if try_acquire_token is None or try_acquire_token == 0:
            self.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip)
        transaction_bytes = self.sync_object_full.tx_count if expected_tx is None else expected_tx
        if self.is_mma_leader_cta:
            self.sync_object_full.arrive_and_expect_tx(
                state.index, transaction_bytes, loc=loc, ip=ip
            )

    @cute.jit
    def producer_commit(
        self,
        state: pipeline.PipelineState,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        pass

    @cute.jit
    def consumer_release(
        self,
        state: pipeline.PipelineState,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        if self.is_mma_leader_cta:
            self.sync_object_empty.arrive(
                state.index, self.consumer_mask, self.cta_group, loc=loc, ip=ip
            )


__all__ = [
    "PipelineTmaUmmaMixedCga",
    "TmaAtomOrPair",
    "bind_executable_tma_load_fields",
    "make_executable_tma_atom",
    "make_runtime_init_pipeline_async",
    "pipeline_init_arrive_mixed_cga",
    "pipeline_init_wait_mixed_cga",
    "tma_multicast_mask",
]
