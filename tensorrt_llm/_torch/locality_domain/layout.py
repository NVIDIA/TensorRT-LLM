# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""locality domain tensor layout metadata.

This module centralizes logical/padded shape and partition-slice reasoning so
module code does not need to duplicate padding arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

NVFP4_WEIGHT_ROW_ALIGNMENT = 32
BF16_PARTITION_ROW_ALIGNMENT = 8


def pad_up(value: int, alignment: int) -> int:
    """Round value up to the next alignment multiple."""
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class PartitionedTensorLayout:
    """Logical/padded layout for a tensor split along one axis."""

    logical_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    partition_axis: int
    num_partitions: int
    axis_name: str = "axis"
    axis_alignment: int = 1
    alignment_name: str = "alignment"

    def __post_init__(self):
        rank = len(self.logical_shape)
        if rank == 0:
            raise ValueError("logical_shape must not be empty")
        if rank != len(self.padded_shape):
            raise ValueError(
                f"logical_shape rank {rank} does not match padded_shape rank "
                f"{len(self.padded_shape)}"
            )

        partition_axis = self.partition_axis
        if partition_axis < 0:
            partition_axis += rank
        if partition_axis < 0 or partition_axis >= rank:
            raise ValueError(
                f"partition_axis {self.partition_axis} is out of bounds for rank {rank}"
            )
        object.__setattr__(self, "partition_axis", partition_axis)

        if self.num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive, got {self.num_partitions}")
        if self.axis_alignment <= 0:
            raise ValueError(f"axis_alignment must be positive, got {self.axis_alignment}")

        for dim, (logical, padded) in enumerate(zip(self.logical_shape, self.padded_shape)):
            if logical <= 0:
                raise ValueError(f"logical_shape[{dim}] must be positive, got {logical}")
            if padded < logical:
                raise ValueError(
                    f"padded_shape[{dim}]={padded} is smaller than logical_shape[{dim}]={logical}"
                )

    @property
    def logical_axis_extent(self) -> int:
        return self.logical_shape[self.partition_axis]

    @property
    def padded_axis_extent(self) -> int:
        return self.padded_shape[self.partition_axis]

    @property
    def is_axis_padding_free(self) -> bool:
        return self.logical_axis_extent == self.padded_axis_extent

    def per_partition_axis_extent(self, *, padded: bool) -> int:
        axis_extent = self.padded_axis_extent if padded else self.logical_axis_extent
        if axis_extent % self.num_partitions != 0:
            shape_kind = "padded" if padded else "logical"
            raise ValueError(
                f"{shape_kind} {self.axis_name}={axis_extent} not divisible by "
                f"num_partitions={self.num_partitions}"
            )
        return axis_extent // self.num_partitions

    def partition_axis_slice(self, partition_id: int, *, padded: bool) -> slice:
        if partition_id < 0 or partition_id >= self.num_partitions:
            raise ValueError(
                f"partition_id={partition_id} out of range for num_partitions={self.num_partitions}"
            )
        extent = self.per_partition_axis_extent(padded=padded)
        return slice(partition_id * extent, (partition_id + 1) * extent)

    def partition_slice(self, partition_id: int, *, padded: bool) -> tuple[slice, ...]:
        slices = [slice(None)] * len(self.logical_shape)
        slices[self.partition_axis] = self.partition_axis_slice(partition_id, padded=padded)
        return tuple(slices)

    def disabled_reason_for_padding_free_split(self) -> str | None:
        """Return why a split that cannot tolerate per-partition padding is invalid."""
        if self.logical_axis_extent % self.num_partitions != 0:
            return (
                f"{self.axis_name}={self.logical_axis_extent} not divisible "
                f"by num_partitions={self.num_partitions}"
            )
        if self.padded_axis_extent % self.num_partitions != 0:
            return (
                f"padded {self.axis_name}={self.padded_axis_extent} not divisible "
                f"by num_partitions={self.num_partitions}"
            )
        if not self.is_axis_padding_free:
            return (
                f"{self.axis_name}={self.logical_axis_extent} not divisible "
                f"by {self.alignment_name}={self.axis_alignment}"
            )
        return None


def make_nvfp4_linear_output_layout(
    out_features: int,
    in_features: int,
    num_partitions: int,
    *,
    padded_out_features: int | None = None,
) -> PartitionedTensorLayout:
    """Build layout metadata for NVFP4 Linear partitioning along output N."""
    if padded_out_features is None:
        padded_out_features = pad_up(out_features, NVFP4_WEIGHT_ROW_ALIGNMENT)
    return PartitionedTensorLayout(
        logical_shape=(out_features, in_features),
        padded_shape=(padded_out_features, in_features),
        partition_axis=0,
        num_partitions=num_partitions,
        axis_name="out_features",
        axis_alignment=NVFP4_WEIGHT_ROW_ALIGNMENT,
        alignment_name="NVFP4 row alignment",
    )


def make_bf16_linear_output_layout(
    out_features: int,
    in_features: int,
    num_partitions: int,
) -> PartitionedTensorLayout:
    """Build padding-free BF16 Linear layout metadata for an output-N split.

    The Rubin BF16 kernel requires each partition's contiguous N dimension to
    be 16-byte aligned. BF16 elements are two bytes, so the full output
    dimension must be a multiple of ``8 * num_partitions`` elements.
    """
    output_row_alignment = BF16_PARTITION_ROW_ALIGNMENT * num_partitions
    return PartitionedTensorLayout(
        logical_shape=(out_features, in_features),
        padded_shape=(
            pad_up(out_features, output_row_alignment),
            in_features,
        ),
        partition_axis=0,
        num_partitions=num_partitions,
        axis_name="out_features",
        axis_alignment=output_row_alignment,
        alignment_name="BF16 locality domain row alignment",
    )
