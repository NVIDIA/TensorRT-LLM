# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent grid-stride and atomic work-ID claim backends."""

from typing import List, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass.cutlass_dsl import Boolean, Int32, Int64, extract_mlir_values, new_from_mlir_values

from .moe_utils import (
    _nanosleep,
    mbarrier_arrive_expect_tx_on_peer,
    store_i32_to_peer_cluster_smem_async,
)


class GridStrideWorkIdState:
    """Register state for one monotonic grid-stride work-ID stream."""

    def __init__(self, next_work_id: Int32, work_id_stride: Int32) -> None:
        self.next_work_id = next_work_id
        self.work_id_stride = work_id_stride

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        values.extend(extract_mlir_values(self.next_work_id))
        values.extend(extract_mlir_values(self.work_id_stride))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "GridStrideWorkIdState":
        next_work_id_value_count = len(extract_mlir_values(self.next_work_id))
        stride_value_count = len(extract_mlir_values(self.work_id_stride))
        expected_value_count = next_work_id_value_count + stride_value_count
        if len(values) != expected_value_count:
            raise ValueError(
                "GridStrideWorkIdState MLIR value count mismatch: "
                f"expected {expected_value_count}, got {len(values)}."
            )
        return type(self)(
            next_work_id=new_from_mlir_values(self.next_work_id, values[:next_work_id_value_count]),
            work_id_stride=new_from_mlir_values(
                self.work_id_stride, values[next_work_id_value_count:]
            ),
        )


class AtomicCounterWorkIdState:
    """Cluster-wide state for one of several atomic work-ID streams."""

    def __init__(
        self,
        counter_pointer: cute.Pointer,
        counter_count: int,
        broadcast_pointer: cute.Pointer,
        is_leader_cta: Boolean,
        cluster_pipeline: pipeline.PipelineAsync,
        producer_state,
        consumer_state,
        cluster_size: int | Int32,
    ) -> None:
        if (
            isinstance(counter_count, bool)
            or not isinstance(counter_count, int)
            or counter_count <= 0
        ):
            raise ValueError("counter_count must be a positive Python int.")
        self.counter_pointer = counter_pointer
        self.counter_count = counter_count
        self.broadcast_pointer = broadcast_pointer
        self.is_leader_cta = is_leader_cta
        self.cluster_pipeline = cluster_pipeline
        self.producer_state = producer_state
        self.consumer_state = consumer_state
        self.cluster_size = cluster_size

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for field in (
            self.counter_pointer,
            self.broadcast_pointer,
            self.is_leader_cta,
            self.producer_state,
            self.consumer_state,
        ):
            values.extend(extract_mlir_values(field))
        if isinstance(self.cluster_size, Int32):
            values.extend(extract_mlir_values(self.cluster_size))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "AtomicCounterWorkIdState":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            result = new_from_mlir_values(
                field, values[value_index : value_index + field_value_count]
            )
            value_index += field_value_count
            return result

        result = type(self)(
            counter_pointer=rebuild(self.counter_pointer),
            counter_count=self.counter_count,
            broadcast_pointer=rebuild(self.broadcast_pointer),
            is_leader_cta=rebuild(self.is_leader_cta),
            cluster_pipeline=self.cluster_pipeline,
            producer_state=rebuild(self.producer_state),
            consumer_state=rebuild(self.consumer_state),
            cluster_size=(
                rebuild(self.cluster_size)
                if isinstance(self.cluster_size, Int32)
                else self.cluster_size
            ),
        )
        if value_index != len(values):
            raise ValueError(
                "AtomicCounterWorkIdState MLIR value count mismatch: "
                f"consumed {value_index}, got {len(values)}."
            )
        return result


class FixedGroupMixedCgaAtomicCounterWorkIdState:
    """Atomic-counter state for fixed groups of fallback clusters."""

    def __init__(
        self,
        atomic_counter_state: AtomicCounterWorkIdState,
        registration_counter_pointer: cute.Pointer,
        group_token_pointer: cute.Pointer,
        split_factor: int,
        fallback_cluster_count: int,
        is_fallback_cluster: Boolean,
        fallback_group_idx: Int32,
        in_group_idx: Int32,
        previous_token: Int64,
        next_generation: Int32,
        claimed_counter_index: Int32,
    ) -> None:
        if isinstance(split_factor, bool) or not isinstance(split_factor, int) or split_factor <= 1:
            raise ValueError("split_factor must be a Python int greater than one.")
        if (
            isinstance(fallback_cluster_count, bool)
            or not isinstance(fallback_cluster_count, int)
            or fallback_cluster_count <= 0
        ):
            raise ValueError("fallback_cluster_count must be a positive Python int.")
        if fallback_cluster_count % split_factor != 0:
            raise ValueError("fallback_cluster_count must be divisible by split_factor.")
        if atomic_counter_state.counter_count > 2:
            raise ValueError("Fixed fallback groups support at most two work-ID streams.")
        self.atomic_counter_state = atomic_counter_state
        self.registration_counter_pointer = registration_counter_pointer
        self.group_token_pointer = group_token_pointer
        self.split_factor = split_factor
        self.fallback_cluster_count = fallback_cluster_count
        self.is_fallback_cluster = is_fallback_cluster
        self.is_preferred_cluster = is_fallback_cluster == Boolean(False)
        self.fallback_group_idx = fallback_group_idx
        self.in_group_idx = in_group_idx
        self.previous_token = previous_token
        self.next_generation = next_generation
        self.claimed_counter_index = claimed_counter_index

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        for field in (
            self.atomic_counter_state,
            self.registration_counter_pointer,
            self.group_token_pointer,
            self.is_fallback_cluster,
            self.fallback_group_idx,
            self.in_group_idx,
            self.previous_token,
            self.next_generation,
            self.claimed_counter_index,
        ):
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(
        self, values: List[ir.Value]
    ) -> "FixedGroupMixedCgaAtomicCounterWorkIdState":
        value_index = 0

        def rebuild(field):
            nonlocal value_index
            field_value_count = len(extract_mlir_values(field))
            result = new_from_mlir_values(
                field, values[value_index : value_index + field_value_count]
            )
            value_index += field_value_count
            return result

        result = type(self)(
            atomic_counter_state=rebuild(self.atomic_counter_state),
            registration_counter_pointer=rebuild(self.registration_counter_pointer),
            group_token_pointer=rebuild(self.group_token_pointer),
            split_factor=self.split_factor,
            fallback_cluster_count=self.fallback_cluster_count,
            is_fallback_cluster=rebuild(self.is_fallback_cluster),
            fallback_group_idx=rebuild(self.fallback_group_idx),
            in_group_idx=rebuild(self.in_group_idx),
            previous_token=rebuild(self.previous_token),
            next_generation=rebuild(self.next_generation),
            claimed_counter_index=rebuild(self.claimed_counter_index),
        )
        if value_index != len(values):
            raise ValueError(
                "FixedGroupMixedCgaAtomicCounterWorkIdState MLIR value count "
                f"mismatch: consumed {value_index}, got {len(values)}."
            )
        return result


@cute.jit
def _claim_grid_stride_work_id(
    work_id_state: GridStrideWorkIdState,
) -> Tuple[Int32, GridStrideWorkIdState]:
    """Claim the next ID from one monotonic grid-stride stream."""
    linear_work_id = work_id_state.next_work_id
    work_id_state.next_work_id = linear_work_id + work_id_state.work_id_stride
    return linear_work_id, work_id_state


@cute.jit
def _claim_atomic_counter_work_id(
    work_id_state: AtomicCounterWorkIdState,
    atomic_counter_index=0,
) -> Tuple[Int32, AtomicCounterWorkIdState]:
    """Claim from one selected counter and broadcast within the cluster."""
    invalid_static_index = isinstance(atomic_counter_index, int) and (
        atomic_counter_index < 0 or atomic_counter_index >= work_id_state.counter_count
    )
    if cutlass.const_expr(invalid_static_index):
        raise ValueError(
            f"atomic_counter_index must be in [0, {work_id_state.counter_count}), "
            f"got {atomic_counter_index}."
        )
    broadcast_tensor = cute.make_tensor(work_id_state.broadcast_pointer, cute.make_layout((1,)))
    cluster_pipeline = work_id_state.cluster_pipeline
    selected_counter_pointer = work_id_state.counter_pointer + Int32(atomic_counter_index)

    if work_id_state.is_leader_cta:
        cluster_pipeline.producer_acquire(work_id_state.producer_state)
        full_barrier_pointer = cluster_pipeline.sync_object_full.get_barrier(
            work_id_state.producer_state.index
        )
        thread_idx, _, _ = cute.arch.thread_idx()
        lane_idx = thread_idx % Int32(32)
        atomic_work_id = Int32(0)
        if lane_idx == Int32(0):
            atomic_work_id = cute.arch.atomic_add(selected_counter_pointer, Int32(1))
        atomic_work_id = cute.arch.shuffle_sync(
            atomic_work_id,
            offset=0,
            mask=0xFFFFFFFF,
            mask_and_clamp=31,
        )
        if lane_idx < Int32(work_id_state.cluster_size):
            store_i32_to_peer_cluster_smem_async(
                work_id_state.broadcast_pointer,
                atomic_work_id,
                full_barrier_pointer,
                lane_idx,
            )
            mbarrier_arrive_expect_tx_on_peer(full_barrier_pointer, Int32(4), lane_idx)
    work_id_state.producer_state.advance()

    cluster_pipeline.consumer_wait(work_id_state.consumer_state)
    linear_work_id = broadcast_tensor[0]
    cute.arch.fence_acq_rel_cta()
    cluster_pipeline.sync_object_empty.arrive(work_id_state.consumer_state.index, Int32(0))
    work_id_state.consumer_state.advance()
    return linear_work_id, work_id_state


@cute.jit
def initialize_fixed_group_mixed_cga_work_id_state(
    work_id_state: FixedGroupMixedCgaAtomicCounterWorkIdState,
) -> FixedGroupMixedCgaAtomicCounterWorkIdState:
    """Register a fallback cluster and broadcast its fixed group indices."""
    atomic_counter_state = work_id_state.atomic_counter_state
    if work_id_state.is_fallback_cluster:
        broadcast_tensor = cute.make_tensor(
            atomic_counter_state.broadcast_pointer, cute.make_layout((1,))
        )
        cluster_pipeline = atomic_counter_state.cluster_pipeline
        if atomic_counter_state.is_leader_cta:
            cluster_pipeline.producer_acquire(atomic_counter_state.producer_state)
            full_barrier_pointer = cluster_pipeline.sync_object_full.get_barrier(
                atomic_counter_state.producer_state.index
            )
            thread_idx, _, _ = cute.arch.thread_idx()
            lane_idx = thread_idx % Int32(32)
            fallback_ordinal = Int32(0)
            if lane_idx == Int32(0):
                fallback_ordinal = cute.arch.atomic_add(
                    work_id_state.registration_counter_pointer,
                    Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
            fallback_ordinal = Int32(
                cute.arch.shuffle_sync(
                    fallback_ordinal,
                    offset=0,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )
            )
            if lane_idx < Int32(atomic_counter_state.cluster_size):
                store_i32_to_peer_cluster_smem_async(
                    atomic_counter_state.broadcast_pointer,
                    fallback_ordinal,
                    full_barrier_pointer,
                    lane_idx,
                )
                mbarrier_arrive_expect_tx_on_peer(full_barrier_pointer, Int32(4), lane_idx)
        atomic_counter_state.producer_state.advance()

        cluster_pipeline.consumer_wait(atomic_counter_state.consumer_state)
        fallback_ordinal = broadcast_tensor[0]
        cute.arch.fence_acq_rel_cta()
        cluster_pipeline.sync_object_empty.arrive(
            atomic_counter_state.consumer_state.index, Int32(0)
        )
        atomic_counter_state.consumer_state.advance()

        fallback_group_idx = fallback_ordinal // Int32(work_id_state.split_factor)
        work_id_state.fallback_group_idx = fallback_group_idx
        work_id_state.in_group_idx = fallback_ordinal - fallback_group_idx * Int32(
            work_id_state.split_factor
        )
    work_id_state.atomic_counter_state = atomic_counter_state
    return work_id_state


@cute.jit
def _claim_fixed_group_fallback_work_id(
    work_id_state: FixedGroupMixedCgaAtomicCounterWorkIdState,
    atomic_counter_index=0,
) -> Tuple[Int32, FixedGroupMixedCgaAtomicCounterWorkIdState]:
    """Claim one ID and hand it to every member of a fallback group."""
    atomic_counter_state = work_id_state.atomic_counter_state
    invalid_static_index = isinstance(atomic_counter_index, int) and (
        atomic_counter_index < 0 or atomic_counter_index >= atomic_counter_state.counter_count
    )
    if cutlass.const_expr(invalid_static_index):
        raise ValueError(
            f"atomic_counter_index must be in [0, {atomic_counter_state.counter_count}), "
            f"got {atomic_counter_index}."
        )

    broadcast_tensor = cute.make_tensor(
        atomic_counter_state.broadcast_pointer, cute.make_layout((1,))
    )
    cluster_pipeline = atomic_counter_state.cluster_pipeline
    selected_counter_pointer = atomic_counter_state.counter_pointer + Int32(atomic_counter_index)

    if atomic_counter_state.is_leader_cta:
        cluster_pipeline.producer_acquire(atomic_counter_state.producer_state)
        full_barrier_pointer = cluster_pipeline.sync_object_full.get_barrier(
            atomic_counter_state.producer_state.index
        )
        thread_idx, _, _ = cute.arch.thread_idx()
        lane_idx = thread_idx % Int32(32)
        group_base_offset = work_id_state.fallback_group_idx * Int32(work_id_state.split_factor)
        group_token_pointer = work_id_state.group_token_pointer + group_base_offset
        claimed_payload = Int32(0)

        if work_id_state.in_group_idx == Int32(0):
            all_members_consumed = Boolean(False)
            while not all_members_consumed:
                observed_token = work_id_state.previous_token
                if lane_idx < Int32(work_id_state.split_factor):
                    observed_token = cute.arch.load(
                        group_token_pointer + lane_idx,
                        Int64,
                        sem="acquire",
                        scope="gpu",
                    )
                lane_is_ready = (lane_idx >= Int32(work_id_state.split_factor)) | (
                    observed_token == work_id_state.previous_token
                )
                ready_mask = Int32(cute.arch.vote_ballot_sync(lane_is_ready))
                all_members_consumed = ready_mask == Int32(-1)
                if not all_members_consumed:
                    _nanosleep(500)

            claimed_work_id = Int32(0)
            if lane_idx == Int32(0):
                claimed_work_id = cute.arch.atomic_add(
                    selected_counter_pointer,
                    Int32(1),
                    sem="relaxed",
                    scope="gpu",
                )
            claimed_work_id = Int32(
                cute.arch.shuffle_sync(
                    claimed_work_id,
                    offset=0,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )
            )
            claimed_payload = claimed_work_id | (Int32(atomic_counter_index) << Int32(31))
            token = (Int64(work_id_state.next_generation) << Int64(32)) | (
                Int64(claimed_payload) & Int64(0xFFFFFFFF)
            )
            if lane_idx == Int32(0):
                cute.arch.store(group_token_pointer, token, sem="relaxed", scope="gpu")
            work_id_state.previous_token = token
            work_id_state.next_generation = work_id_state.next_generation + Int32(1)
        else:
            token = work_id_state.previous_token
            while token == work_id_state.previous_token:
                token_high = Int32(0)
                token_low = Int32(0)
                if lane_idx == Int32(0):
                    observed_token = cute.arch.load(
                        group_token_pointer, Int64, sem="relaxed", scope="gpu"
                    )
                    token_high = Int32(observed_token >> Int64(32))
                    token_low = Int32(observed_token & Int64(0xFFFFFFFF))
                token_high = Int32(
                    cute.arch.shuffle_sync(
                        token_high,
                        offset=0,
                        mask=0xFFFFFFFF,
                        mask_and_clamp=31,
                    )
                )
                token_low = Int32(
                    cute.arch.shuffle_sync(
                        token_low,
                        offset=0,
                        mask=0xFFFFFFFF,
                        mask_and_clamp=31,
                    )
                )
                token = (Int64(token_high) << Int64(32)) | (Int64(token_low) & Int64(0xFFFFFFFF))
                if token == work_id_state.previous_token:
                    _nanosleep(500)
            claimed_payload = Int32(token & Int64(0xFFFFFFFF))
            if lane_idx == Int32(0):
                cute.arch.store(
                    group_token_pointer + work_id_state.in_group_idx,
                    token,
                    sem="relaxed",
                    scope="gpu",
                )
            work_id_state.previous_token = token

        if lane_idx < Int32(atomic_counter_state.cluster_size):
            store_i32_to_peer_cluster_smem_async(
                atomic_counter_state.broadcast_pointer,
                claimed_payload,
                full_barrier_pointer,
                lane_idx,
            )
            mbarrier_arrive_expect_tx_on_peer(full_barrier_pointer, Int32(4), lane_idx)
    atomic_counter_state.producer_state.advance()

    cluster_pipeline.consumer_wait(atomic_counter_state.consumer_state)
    claimed_payload = broadcast_tensor[0]
    cute.arch.fence_acq_rel_cta()
    cluster_pipeline.sync_object_empty.arrive(atomic_counter_state.consumer_state.index, Int32(0))
    atomic_counter_state.consumer_state.advance()
    work_id_state.atomic_counter_state = atomic_counter_state
    work_id_state.claimed_counter_index = (claimed_payload >> Int32(31)) & Int32(1)
    linear_work_id = claimed_payload & Int32(0x7FFFFFFF)
    return linear_work_id, work_id_state


@cute.jit
def _claim_fixed_group_mixed_cga_work_id(
    work_id_state: FixedGroupMixedCgaAtomicCounterWorkIdState,
    atomic_counter_index=0,
) -> Tuple[Int32, FixedGroupMixedCgaAtomicCounterWorkIdState]:
    """Claim directly or through the preferred cluster's fallback group."""
    linear_work_id = Int32(0)
    if work_id_state.is_preferred_cluster:
        linear_work_id, atomic_counter_state = _claim_atomic_counter_work_id(
            work_id_state.atomic_counter_state, atomic_counter_index
        )
        work_id_state.atomic_counter_state = atomic_counter_state
        work_id_state.claimed_counter_index = Int32(atomic_counter_index)
    else:
        linear_work_id, work_id_state = _claim_fixed_group_fallback_work_id(
            work_id_state, atomic_counter_index
        )
    return linear_work_id, work_id_state


@cute.jit
def claim_work_id(work_id_state, atomic_counter_index=0):
    """Claim the next work ID using the backend encoded by the state type."""
    if cutlass.const_expr(isinstance(work_id_state, GridStrideWorkIdState)):
        return _claim_grid_stride_work_id(work_id_state)
    if cutlass.const_expr(isinstance(work_id_state, AtomicCounterWorkIdState)):
        return _claim_atomic_counter_work_id(work_id_state, atomic_counter_index)
    if cutlass.const_expr(isinstance(work_id_state, FixedGroupMixedCgaAtomicCounterWorkIdState)):
        return _claim_fixed_group_mixed_cga_work_id(work_id_state, atomic_counter_index)
    raise TypeError(f"Unsupported work-ID state: {type(work_id_state).__name__}.")


__all__ = [
    "AtomicCounterWorkIdState",
    "FixedGroupMixedCgaAtomicCounterWorkIdState",
    "GridStrideWorkIdState",
    "claim_work_id",
    "initialize_fixed_group_mixed_cga_work_id_state",
]
