# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host validation and topology for MLA's parallel standalone reducer."""

from __future__ import annotations

from dataclasses import dataclass


_MAX_SPLITS_KV = 128
_TARGET_SPLITS_PER_RANK = 8
_PARALLEL_REDUCER_MIN_SPLITS_PER_RANK = 32
_PARALLEL_REDUCER_MAX_CLUSTER_WAVES = 4
_SUPPORTED_CLUSTER_SIZES = (1, 2, 4, 8, 16)
_Q128_OUTPUT_ELEMENTS_PER_ROW = 512
_OUTPUT_ELEMENTS_PER_WORK_UNIT = 1024
# Some workspace layout products remain 32-bit. Keep the parallel reducer below
# that boundary until every producer and tensor-layout stride is 64-bit safe.
_MAX_PARTIAL_O_WORKSPACE_ELEMENTS = 2**31 - 1


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _validate_splits_kv(splits_kv: int) -> None:
    if isinstance(splits_kv, bool) or not isinstance(splits_kv, int):
        raise TypeError("splits_kv must be an integer")
    if not 1 <= splits_kv <= _MAX_SPLITS_KV:
        raise ValueError(f"splits_kv must be in [1, {_MAX_SPLITS_KV}]")


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_cluster_size(max_cluster_size: int) -> None:
    if max_cluster_size not in _SUPPORTED_CLUSTER_SIZES:
        raise ValueError(
            f"max_cluster_size must be one of {set(_SUPPORTED_CLUSTER_SIZES)}"
        )


def choose_q64_parallel_reducer_cluster_size(
    splits_kv: int,
    *,
    base_clusters: int,
    sm_count: int,
    max_cluster_size: int = 16,
) -> int:
    """Choose Q64 G from actual split work and a four-wave cluster bound.

    Keep at least one warp (32) of actual splits on every rank to amortize
    DSMEM publication and merging. A G>1 candidate is accepted only when the
    clustered launch fits in four waves; G1 remains the unrestricted fallback.
    With S<=128, the per-rank work bound limits production Q64 to G<=4.
    """

    _validate_splits_kv(splits_kv)
    _validate_positive_int(base_clusters, "base_clusters")
    _validate_positive_int(sm_count, "sm_count")
    _validate_cluster_size(max_cluster_size)

    split_limit = max(1, splits_kv // _PARALLEL_REDUCER_MIN_SPLITS_PER_RANK)
    split_cluster_size = 1 << (split_limit.bit_length() - 1)
    split_cluster_size = min(split_cluster_size, max_cluster_size)

    for cluster_size in reversed(_SUPPORTED_CLUSTER_SIZES):
        if cluster_size > split_cluster_size:
            continue
        clusters_per_wave = sm_count // cluster_size
        if clusters_per_wave == 0:
            continue
        waves = (base_clusters + clusters_per_wave - 1) // clusters_per_wave
        if waves <= _PARALLEL_REDUCER_MAX_CLUSTER_WAVES:
            return cluster_size
    return 1


def validate_parallel_reduction_workspace(
    *,
    batch_size: int,
    num_heads_q: int,
    seq_len_q: int,
    splits_kv: int,
    head_dim: int,
) -> int:
    """Validate and return the normalized partial-O workspace element count."""

    _validate_positive_int(batch_size, "batch_size")
    _validate_positive_int(num_heads_q, "num_heads_q")
    _validate_positive_int(seq_len_q, "seq_len_q")
    _validate_splits_kv(splits_kv)
    _validate_positive_int(head_dim, "head_dim")
    if splits_kv == 1:
        raise ValueError("parallel reduction requires splits_kv in [2, 128]")

    workspace_elements = batch_size * num_heads_q * seq_len_q * splits_kv * head_dim
    if workspace_elements > _MAX_PARTIAL_O_WORKSPACE_ELEMENTS:
        raise ValueError(
            "parallel reduction requires fewer than 2^31 partial-O workspace "
            "elements until every layout stride is qualified as 64-bit safe"
        )
    return workspace_elements


@dataclass(frozen=True)
class ParallelReductionTopology:
    """Padded split slots distributed uniformly across one CTA cluster."""

    actual_splits: int
    padded_splits: int
    cluster_size: int
    slots_per_rank: int
    interleaved: bool = False

    @property
    def padding_slots(self) -> int:
        return self.cluster_size * self.slots_per_rank - self.actual_splits

    def split_for_slot(self, rank: int, slot: int) -> int | None:
        """Map a rank-local slot to a split, or ``None`` for padding."""

        if not 0 <= rank < self.cluster_size:
            raise ValueError(f"rank must be in [0, {self.cluster_size})")
        if not 0 <= slot < self.slots_per_rank:
            raise ValueError(f"slot must be in [0, {self.slots_per_rank})")

        split = (
            slot * self.cluster_size + rank
            if self.interleaved
            else rank * self.slots_per_rank + slot
        )
        return split if split < self.actual_splits else None


def make_balanced_parallel_reduction_topology(
    splits_kv: int,
    *,
    cluster_size: int,
) -> ParallelReductionTopology | None:
    """Distribute actual splits cyclically with fewer than G padding slots."""

    _validate_splits_kv(splits_kv)
    _validate_cluster_size(cluster_size)
    if splits_kv == 1:
        return None
    slots_per_rank = (splits_kv + cluster_size - 1) // cluster_size
    capacity_splits = cluster_size * slots_per_rank
    return ParallelReductionTopology(
        actual_splits=splits_kv,
        padded_splits=capacity_splits,
        cluster_size=cluster_size,
        slots_per_rank=slots_per_rank,
        interleaved=True,
    )


def make_parallel_reduction_topology(
    splits_kv: int,
    *,
    max_cluster_size: int = 16,
) -> ParallelReductionTopology | None:
    """Build the S2..S128 topology, returning ``None`` for the S1 bypass.

    S2..S16 use exact-capacity G1 so small reductions avoid both padding and
    cluster exchange. S17+ retain the power-of-two split capacity and
    split-derived power-of-two cluster topology. Slots beyond ``actual_splits``
    must skip loads and computation.
    """

    _validate_splits_kv(splits_kv)
    _validate_cluster_size(max_cluster_size)
    if splits_kv == 1:
        return None
    if splits_kv <= 16:
        return ParallelReductionTopology(
            actual_splits=splits_kv,
            padded_splits=splits_kv,
            cluster_size=1,
            slots_per_rank=splits_kv,
        )

    padded_splits = _next_power_of_two(splits_kv)
    cluster_size = min(
        max_cluster_size,
        _next_power_of_two(
            (splits_kv + _TARGET_SPLITS_PER_RANK - 1) // _TARGET_SPLITS_PER_RANK
        ),
    )
    return ParallelReductionTopology(
        actual_splits=splits_kv,
        padded_splits=padded_splits,
        cluster_size=cluster_size,
        slots_per_rank=padded_splits // cluster_size,
    )


def make_q128_wave_limited_parallel_reduction_topology(
    splits_kv: int,
    *,
    logical_rows: int,
    physical_sm_count: int,
    max_cluster_size: int = 16,
) -> ParallelReductionTopology | None:
    """Build Q128/D512 topology targeting four output-work waves.

    One base work unit is 1,024 output elements. Start from the split-derived
    topology, then reduce G until those work units fit within a four-wave
    pressure target, or until G1 is reached. This is a work proxy rather than a
    literal count of launched CTA waves. If G changes, recompute the minimum
    contiguous split capacity supported by the retained ranks.
    """

    _validate_splits_kv(splits_kv)
    _validate_positive_int(logical_rows, "logical_rows")
    _validate_positive_int(physical_sm_count, "physical_sm_count")
    _validate_cluster_size(max_cluster_size)

    topology = make_parallel_reduction_topology(
        splits_kv,
        max_cluster_size=max_cluster_size,
    )
    if topology is None:
        return None
    base_work_units = (
        logical_rows * _Q128_OUTPUT_ELEMENTS_PER_ROW
        + _OUTPUT_ELEMENTS_PER_WORK_UNIT
        - 1
    ) // _OUTPUT_ELEMENTS_PER_WORK_UNIT
    cluster_size = topology.cluster_size
    while cluster_size > 1:
        work_units_per_wave = physical_sm_count // cluster_size
        if work_units_per_wave == 0:
            cluster_size //= 2
            continue
        pressure_waves = (
            base_work_units + work_units_per_wave - 1
        ) // work_units_per_wave
        if pressure_waves <= _PARALLEL_REDUCER_MAX_CLUSTER_WAVES:
            break
        cluster_size //= 2

    if cluster_size == topology.cluster_size:
        return topology
    slots_per_rank = (splits_kv + cluster_size - 1) // cluster_size
    return ParallelReductionTopology(
        actual_splits=splits_kv,
        padded_splits=cluster_size * slots_per_rank,
        cluster_size=cluster_size,
        slots_per_rank=slots_per_rank,
    )
