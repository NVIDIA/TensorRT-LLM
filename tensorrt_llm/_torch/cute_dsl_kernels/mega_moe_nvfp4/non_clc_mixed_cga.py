# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preferred/fallback cluster scheduling without hardware CLC."""

import dataclasses
from typing import List, Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass.cutlass_dsl import Boolean, Int32, Int64, extract_mlir_values, new_from_mlir_values

from .work_id_claim import (
    AtomicCounterWorkIdState,
    FixedGroupMixedCgaAtomicCounterWorkIdState,
    GridStrideWorkIdState,
    claim_work_id,
    initialize_fixed_group_mixed_cga_work_id_state,
)


@dataclasses.dataclass(frozen=True)
class NonClcMixedCgaConfig:
    """Static launch geometry for non-CLC mixed-CGA schedulers."""

    preferred_cluster_shape: Tuple[int, int]
    fallback_cluster_shape: Optional[Tuple[int, int]]
    launch_cluster_count: Optional[int]
    preferred_cluster_count: Optional[int]
    fallback_cluster_count: Optional[int]
    mn_split_factors: Tuple[int, int] = dataclasses.field(init=False)
    split_factor: int = dataclasses.field(init=False)
    launch_cluster_cnt_merge_as_preferred: int = dataclasses.field(init=False)
    total_cta_cnt: int = dataclasses.field(init=False)
    is_mixed: bool = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._validate_shape(self.preferred_cluster_shape, "preferred_cluster_shape")
        preferred_cluster_size = self._shape_size(self.preferred_cluster_shape)
        if self.fallback_cluster_shape is not None:
            self._validate_shape(self.fallback_cluster_shape, "fallback_cluster_shape")
            has_no_fallback_clusters = (
                isinstance(self.fallback_cluster_count, int)
                and not isinstance(self.fallback_cluster_count, bool)
                and self.fallback_cluster_count == 0
            )
            if (
                self.fallback_cluster_shape == self.preferred_cluster_shape
                or has_no_fallback_clusters
            ):
                object.__setattr__(self, "fallback_cluster_shape", None)
                object.__setattr__(self, "preferred_cluster_count", None)
                object.__setattr__(self, "fallback_cluster_count", None)

        if self.fallback_cluster_shape is None:
            if not self._is_positive_int(self.launch_cluster_count):
                raise ValueError(
                    "launch_cluster_count must be positive when fallback_cluster_shape is absent."
                )
            mn_split_factors = (1, 1)
            split_factor = 1
            launch_cluster_cnt_merge_as_preferred = self.launch_cluster_count
            is_mixed = False
        else:
            if len(self.fallback_cluster_shape) != len(self.preferred_cluster_shape):
                raise ValueError("preferred and fallback cluster shapes must have equal rank.")
            if not self._is_positive_int(self.preferred_cluster_count):
                raise ValueError(
                    "preferred_cluster_count must be positive when "
                    "fallback_cluster_shape is present."
                )
            if (
                isinstance(self.fallback_cluster_count, bool)
                or not isinstance(self.fallback_cluster_count, int)
                or self.fallback_cluster_count < 0
            ):
                raise ValueError(
                    "fallback_cluster_count must be a non-negative Python "
                    "int when fallback_cluster_shape is present."
                )

            if any(
                preferred_dimension % fallback_dimension != 0
                for preferred_dimension, fallback_dimension in zip(
                    self.preferred_cluster_shape, self.fallback_cluster_shape
                )
            ):
                raise ValueError(
                    "Every preferred cluster dimension must be divisible by its fallback dimension."
                )
            mn_split_factors = (
                self.preferred_cluster_shape[0] // self.fallback_cluster_shape[0],
                self.preferred_cluster_shape[1] // self.fallback_cluster_shape[1],
            )
            split_factor = self._shape_size(mn_split_factors)
            is_mixed = self.fallback_cluster_shape != self.preferred_cluster_shape
            if is_mixed:
                if split_factor > 16 or split_factor & (split_factor - 1):
                    raise ValueError(
                        "The preferred/fallback cluster split factor must be "
                        "a power of two at most 16."
                    )
                if self.fallback_cluster_count % split_factor != 0:
                    raise ValueError(
                        "fallback_cluster_count must be divisible by the "
                        "preferred/fallback cluster split factor."
                    )
                launch_cluster_cnt_merge_as_preferred = (
                    self.preferred_cluster_count + self.fallback_cluster_count // split_factor
                )
            else:
                launch_cluster_cnt_merge_as_preferred = (
                    self.preferred_cluster_count + self.fallback_cluster_count
                )

        if launch_cluster_cnt_merge_as_preferred <= 0:
            raise ValueError(
                "The resolved launch must contain at least one cluster merged as preferred."
            )
        object.__setattr__(self, "mn_split_factors", mn_split_factors)
        object.__setattr__(self, "split_factor", split_factor)
        object.__setattr__(
            self, "launch_cluster_cnt_merge_as_preferred", launch_cluster_cnt_merge_as_preferred
        )
        object.__setattr__(
            self, "total_cta_cnt", launch_cluster_cnt_merge_as_preferred * preferred_cluster_size
        )
        object.__setattr__(self, "is_mixed", is_mixed)

    @staticmethod
    def _shape_size(shape: Tuple[int, int]) -> int:
        result = 1
        for dimension in shape:
            result *= dimension
        return result

    @staticmethod
    def _is_positive_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    @classmethod
    def _validate_shape(cls, shape: Tuple[int, int], field_name: str) -> None:
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise TypeError(f"{field_name} must be a two-dimensional tuple.")
        if not all(cls._is_positive_int(dimension) for dimension in shape):
            raise ValueError(f"{field_name} dimensions must be positive Python ints.")


class NonClcMixedCgaSchedulerWorker:
    """Map preferred and fallback clusters onto canonical work-ID streams."""

    def __init__(
        self,
        *,
        config: NonClcMixedCgaConfig,
        work_id_mode: Literal["grid_stride", "atomic_counter"],
        stream_count: int,
    ) -> None:
        if work_id_mode not in ("grid_stride", "atomic_counter"):
            raise ValueError(
                "Non-CLC mixed-CGA scheduling supports grid_stride or atomic_counter work IDs."
            )
        if isinstance(stream_count, bool) or not isinstance(stream_count, int) or stream_count <= 0:
            raise ValueError("stream_count must be a positive Python int.")
        if stream_count > 2:
            raise ValueError("Mixed-CGA scheduling supports at most two streams.")
        self.config = config
        self.work_id_mode = work_id_mode
        self.stream_count = stream_count

    @cute.jit
    def assign_device_members(
        self,
        *,
        is_fallback_cluster: Optional[Boolean],
        block_idx: Tuple,
        counter_pointer=None,
        registration_counter_pointer=None,
        group_token_pointer=None,
        broadcast_pointer=None,
        cluster_pipeline=None,
    ) -> None:
        """Bind one active cluster to the configured work-ID backend."""
        if cutlass.const_expr(self.config.is_mixed and is_fallback_cluster is None):
            raise ValueError("is_fallback_cluster is required for a mixed-CGA launch.")

        active_cluster_m = self.config.preferred_cluster_shape[0]
        active_cluster_n = self.config.preferred_cluster_shape[1]
        if cutlass.const_expr(self.config.is_mixed):
            if is_fallback_cluster:
                active_cluster_m = Int32(self.config.fallback_cluster_shape[0])
                active_cluster_n = Int32(self.config.fallback_cluster_shape[1])

        cta_coord_in_active_cluster = (
            Int32(block_idx[0]) % Int32(active_cluster_m),
            Int32(block_idx[1]) % Int32(active_cluster_n),
            Int32(0),
        )
        cta_coord_in_preferred_cluster = cta_coord_in_active_cluster

        if cutlass.const_expr(self.config.is_mixed and self.work_id_mode == "grid_stride"):
            if is_fallback_cluster:
                flattened_index = Int32(0)
                dimension_stride = 1
                for dimension_idx in cutlass.range_constexpr(len(self.config.mn_split_factors)):
                    preferred_dimension = self.config.preferred_cluster_shape[dimension_idx]
                    fallback_dimension = self.config.fallback_cluster_shape[dimension_idx]
                    inner_coordinate = (
                        Int32(block_idx[dimension_idx]) % Int32(preferred_dimension)
                    ) // Int32(fallback_dimension)
                    flattened_index = flattened_index + inner_coordinate * Int32(dimension_stride)
                    dimension_stride *= self.config.mn_split_factors[dimension_idx]
                cta_coord_in_preferred_cluster = self._preferred_cluster_cta_coord(
                    cta_coord_in_active_cluster, flattened_index
                )
        self.cta_coord_in_preferred_cluster = cta_coord_in_preferred_cluster

        if cutlass.const_expr(self.work_id_mode == "atomic_counter"):
            if cutlass.const_expr(counter_pointer is None or broadcast_pointer is None):
                raise ValueError("atomic_counter requires counter and broadcast pointers.")
            active_cluster_size = active_cluster_m * active_cluster_n
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            atomic_counter_state = AtomicCounterWorkIdState(
                counter_pointer=counter_pointer,
                counter_count=self.stream_count,
                broadcast_pointer=broadcast_pointer,
                is_leader_cta=(cta_coord_in_active_cluster[0] + cta_coord_in_active_cluster[1])
                == Int32(0),
                cluster_pipeline=cluster_pipeline,
                producer_state=producer_state,
                consumer_state=consumer_state,
                cluster_size=active_cluster_size,
            )
            if cutlass.const_expr(self.config.is_mixed):
                if cutlass.const_expr(
                    registration_counter_pointer is None or group_token_pointer is None
                ):
                    raise ValueError(
                        "mixed atomic_counter scheduling requires fallback "
                        "registration and group-token pointers."
                    )
                self._cta_coord_in_active_cluster = cta_coord_in_active_cluster
                self._work_id_state = FixedGroupMixedCgaAtomicCounterWorkIdState(
                    atomic_counter_state=atomic_counter_state,
                    registration_counter_pointer=registration_counter_pointer,
                    group_token_pointer=group_token_pointer,
                    split_factor=self.config.split_factor,
                    fallback_cluster_count=self.config.fallback_cluster_count,
                    is_fallback_cluster=is_fallback_cluster,
                    fallback_group_idx=Int32(0),
                    in_group_idx=Int32(0),
                    previous_token=Int64(0),
                    next_generation=Int32(1),
                    claimed_counter_index=Int32(0),
                )
            else:
                self._work_id_state = atomic_counter_state
        else:
            self._work_id_state = GridStrideWorkIdState(
                next_work_id=Int32(block_idx[2]),
                work_id_stride=Int32(self.config.launch_cluster_cnt_merge_as_preferred),
            )
        self.claimed_stream_index = Int32(0)

    def set_cluster_pipeline(self, cluster_pipeline) -> None:
        """Attach the cluster pipeline after legacy prologue construction."""
        if self.work_id_mode != "atomic_counter":
            return
        if isinstance(self._work_id_state, FixedGroupMixedCgaAtomicCounterWorkIdState):
            self._work_id_state.atomic_counter_state.cluster_pipeline = cluster_pipeline
        else:
            self._work_id_state.cluster_pipeline = cluster_pipeline

    @property
    def active_cluster_size(self):
        """Return the physical active-cluster CTA count."""
        if self.work_id_mode != "atomic_counter":
            return None
        if isinstance(self._work_id_state, FixedGroupMixedCgaAtomicCounterWorkIdState):
            return self._work_id_state.atomic_counter_state.cluster_size
        return self._work_id_state.cluster_size

    @cute.jit
    def initialize_fallback_group(self) -> None:
        """Register one fallback cluster with its fixed logical group."""
        if cutlass.const_expr(self.config.is_mixed and self.work_id_mode == "atomic_counter"):
            self._work_id_state = initialize_fixed_group_mixed_cga_work_id_state(
                self._work_id_state
            )
            if self._work_id_state.is_fallback_cluster:
                self.cta_coord_in_preferred_cluster = self._preferred_cluster_cta_coord(
                    self._cta_coord_in_active_cluster, self._work_id_state.in_group_idx
                )

    @cute.jit
    def _preferred_cluster_cta_coord(
        self,
        cta_coord_in_active_cluster: cute.Coord,
        inner_cluster_idx: Int32,
    ) -> cute.Coord:
        inner_cluster_m = inner_cluster_idx % Int32(self.config.mn_split_factors[0])
        inner_cluster_n = (inner_cluster_idx // Int32(self.config.mn_split_factors[0])) % Int32(
            self.config.mn_split_factors[1]
        )
        return (
            cta_coord_in_active_cluster[0]
            + inner_cluster_m * Int32(self.config.fallback_cluster_shape[0]),
            cta_coord_in_active_cluster[1]
            + inner_cluster_n * Int32(self.config.fallback_cluster_shape[1]),
            Int32(0),
        )

    @cute.jit
    def claim_next_work(self, stream_index=0) -> Int32:
        """Claim a canonical ID and update the preferred CTA coordinate."""
        claimed_work_id, self._work_id_state = claim_work_id(
            self._work_id_state, atomic_counter_index=stream_index
        )
        if cutlass.const_expr(self.config.is_mixed and self.work_id_mode == "atomic_counter"):
            cta_coord_in_preferred_cluster = self._cta_coord_in_active_cluster
            if self._work_id_state.is_fallback_cluster:
                cta_coord_in_preferred_cluster = self._preferred_cluster_cta_coord(
                    self._cta_coord_in_active_cluster, self._work_id_state.in_group_idx
                )
            self.cta_coord_in_preferred_cluster = cta_coord_in_preferred_cluster
            self.claimed_stream_index = self._work_id_state.claimed_counter_index
        else:
            self.claimed_stream_index = Int32(stream_index)
        return claimed_work_id

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values: List[ir.Value] = []
        values.extend(extract_mlir_values(self._work_id_state))
        values.extend(extract_mlir_values(self.cta_coord_in_preferred_cluster))
        values.extend(extract_mlir_values(self.claimed_stream_index))
        if self.config.is_mixed and self.work_id_mode == "atomic_counter":
            values.extend(extract_mlir_values(self._cta_coord_in_active_cluster))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "NonClcMixedCgaSchedulerWorker":
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
            config=self.config,
            work_id_mode=self.work_id_mode,
            stream_count=self.stream_count,
        )
        result._work_id_state = rebuild(self._work_id_state)
        result.cta_coord_in_preferred_cluster = rebuild(self.cta_coord_in_preferred_cluster)
        result.claimed_stream_index = rebuild(self.claimed_stream_index)
        if self.config.is_mixed and self.work_id_mode == "atomic_counter":
            result._cta_coord_in_active_cluster = rebuild(self._cta_coord_in_active_cluster)
        if value_index != len(values):
            raise ValueError(
                "NonClcMixedCgaSchedulerWorker MLIR value count mismatch: "
                f"consumed {value_index}, got {len(values)}."
            )
        return result


__all__ = ["NonClcMixedCgaConfig", "NonClcMixedCgaSchedulerWorker"]
