# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative GMEM workspace shared by host layout and device access."""

import dataclasses
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64

from .utils import (
    ceil_div,
    cosize_from_shape_stride_tuples,
    flatten_shape_stride,
    is_nested_shape,
    is_power_of_two,
    ordered_stride,
    round_up,
    row_major_stride,
    validate_static_integer_tuple,
)

BufferResetAttr = Literal["data", "zero_on_first_allocate", "tail_reset"]
BufferSpace = Literal["local", "shared"]

_reset_order = {"tail_reset": 0, "zero_on_first_allocate": 1, "data": 2}


def _footprint_from_shape_stride(shape: Tuple, stride: Tuple) -> int:
    """Elements a region must own, as opposed to the ones its layout can address."""
    leaf_pairs = flatten_shape_stride(shape, stride) if shape else []
    claimed = max((size * step for size, step in leaf_pairs), default=1)
    # Layouts whose leaves overlap can address past what any single leaf tiles,
    # so the cosize stays a floor.
    return int(max(claimed, cosize_from_shape_stride_tuples(shape, stride)))


@dataclasses.dataclass(frozen=True)
class DeviceRegion:
    """One typed region in a local or symmetric GMEM workspace."""

    name: str
    dtype: Type[cutlass.Numeric]
    shape: Tuple
    buffer_space: BufferSpace
    stride: Optional[Tuple] = None
    mem_order: Optional[Tuple[int, ...]] = None
    byte_alignment: int = 16
    reset: BufferResetAttr = "data"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A workspace region needs a non-empty name.")
        if self.buffer_space not in ("local", "shared"):
            raise ValueError(f"Invalid buffer space {self.buffer_space!r}.")
        if self.reset not in _reset_order:
            raise ValueError(f"Invalid reset policy {self.reset!r}.")
        if not is_power_of_two(self.byte_alignment):
            raise ValueError(f"Region {self.name!r} alignment must be a positive power of two.")
        validate_static_integer_tuple(self.shape, field_name=f"{self.name}.shape")
        if self.stride is not None and self.mem_order is not None:
            raise ValueError(f"Region {self.name!r} accepts either stride or mem_order, not both.")
        if self.stride is None and self.mem_order is None:
            if is_nested_shape(self.shape) or len(self.shape) != 1:
                raise ValueError(f"Region {self.name!r} needs an explicit layout.")
            object.__setattr__(self, "stride", row_major_stride(self.shape))
        if is_nested_shape(self.shape) and self.mem_order is not None:
            raise ValueError(f"Nested region {self.name!r} needs an explicit stride.")
        if self.stride is not None:
            validate_static_integer_tuple(self.stride, field_name=f"{self.name}.stride")
        if self.mem_order is not None:
            validate_static_integer_tuple(self.mem_order, field_name=f"{self.name}.mem_order")
            expected = tuple(range(len(self.shape)))
            if tuple(sorted(self.mem_order)) != expected:
                raise ValueError(
                    f"Region {self.name!r} mem_order must be a permutation of {expected}, got {self.mem_order}."
                )
            resolved_stride, _ = ordered_stride(self.shape, self.mem_order)
            object.__setattr__(self, "stride", resolved_stride)


class DeviceWorkspace:
    """Single source of truth for GMEM region layout and device pointer derivation."""

    def __init__(self) -> None:
        self._registered: Dict[BufferSpace, List[DeviceRegion]] = {"local": [], "shared": []}
        self._region_by_name: Dict[str, DeviceRegion] = {}
        self._offset: Dict[str, int] = {}
        self._stride: Dict[str, Tuple] = {}
        self._cosize: Dict[str, int] = {}
        self._nbytes: Dict[str, int] = {}
        self._byte_alignment: Dict[str, int] = {}
        self._total: Dict[BufferSpace, int] = {"local": 0, "shared": 0}
        self._zero_leading: Dict[BufferSpace, int] = {"local": 0, "shared": 0}
        self._tail_leading: Dict[BufferSpace, int] = {"local": 0, "shared": 0}
        self._base: Dict[BufferSpace, Any] = {"local": None, "shared": None}
        self._finalized = False

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "DeviceWorkspace":
        return self

    @property
    def finalized(self) -> bool:
        return self._finalized

    def register(
        self,
        name: str,
        dtype: Type[cutlass.Numeric],
        shape: Tuple,
        *,
        buffer_space: BufferSpace,
        stride: Optional[Tuple] = None,
        mem_order: Optional[Tuple[int, ...]] = None,
        byte_alignment: int = 16,
        reset: BufferResetAttr = "data",
    ) -> None:
        if self._finalized:
            raise RuntimeError("Cannot register a region after finalize().")
        if name in self._region_by_name or any(
            region.name == name for regions in self._registered.values() for region in regions
        ):
            raise ValueError(f"Duplicate workspace region {name!r}.")
        self._registered[buffer_space].append(
            DeviceRegion(
                name=name,
                dtype=dtype,
                shape=shape,
                buffer_space=buffer_space,
                stride=stride,
                mem_order=mem_order,
                byte_alignment=byte_alignment,
                reset=reset,
            )
        )

    def finalize(self) -> None:
        if self._finalized:
            raise RuntimeError("DeviceWorkspace.finalize() may only be called once.")
        for buffer_space in ("local", "shared"):
            ordered = sorted(
                self._registered[buffer_space], key=lambda region: _reset_order[region.reset]
            )
            cursor = 0
            for region_index, region in enumerate(ordered):
                stride = region.stride
                if stride is None:
                    raise RuntimeError(f"Region {region.name!r} stride was not resolved.")
                cosize = cosize_from_shape_stride_tuples(region.shape, stride)
                nbytes = (
                    _footprint_from_shape_stride(region.shape, stride) * int(region.dtype.width) + 7
                ) // 8
                if region.reset == "tail_reset" and (
                    region_index == 0 or ordered[region_index - 1].reset != "tail_reset"
                ):
                    cursor = round_up(cursor, 16)
                byte_alignment = region.byte_alignment
                cursor = round_up(cursor, byte_alignment)
                self._region_by_name[region.name] = region
                self._offset[region.name] = cursor
                self._stride[region.name] = stride
                self._cosize[region.name] = cosize
                self._nbytes[region.name] = nbytes
                self._byte_alignment[region.name] = byte_alignment
                cursor += nbytes
                is_last_tail_reset_region = region.reset == "tail_reset" and (
                    region_index + 1 == len(ordered)
                    or ordered[region_index + 1].reset != "tail_reset"
                )
                if is_last_tail_reset_region:
                    cursor = round_up(cursor, 16)
                if region.reset != "data":
                    self._zero_leading[buffer_space] = cursor
                if is_last_tail_reset_region:
                    self._tail_leading[buffer_space] = cursor
            self._total[buffer_space] = round_up(cursor, 16)
        self._finalized = True

    def regions(self, buffer_space: BufferSpace) -> Tuple[DeviceRegion, ...]:
        return tuple(self._registered[buffer_space])

    def region(self, name: str) -> DeviceRegion:
        self._require_finalized()
        return self._region_by_name[name]

    def offset(self, name: str) -> int:
        self._require_finalized()
        return self._offset[name]

    def stride(self, name: str) -> Tuple:
        self._require_finalized()
        return self._stride[name]

    def cosize(self, name: str) -> int:
        self._require_finalized()
        return self._cosize[name]

    def nbytes(self, name: str) -> int:
        self._require_finalized()
        return self._nbytes[name]

    def byte_alignment(self, name: str) -> int:
        self._require_finalized()
        return self._byte_alignment[name]

    def total_bytes(self, buffer_space: BufferSpace) -> int:
        self._require_finalized()
        return self._total[buffer_space]

    @property
    def local_and_shared_bytes(self) -> Tuple[int, int]:
        self._require_finalized()
        return self._total["local"], self._total["shared"]

    def zero_on_allocate_bytes(self, buffer_space: BufferSpace) -> int:
        self._require_finalized()
        return self._zero_leading[buffer_space]

    def tail_reset_bytes(self, buffer_space: BufferSpace) -> int:
        self._require_finalized()
        return self._tail_leading[buffer_space]

    @property
    def require_zero_workspace_leading_bytes(self) -> Tuple[int, int]:
        self._require_finalized()
        return self._zero_leading["local"], self._zero_leading["shared"]

    @cute.jit
    def assign_device_members(
        self, local_workspace: cute.Pointer, shared_workspace: Optional[cute.Pointer] = None
    ) -> None:
        self._base["local"] = local_workspace
        self._base["shared"] = shared_workspace

    def remove_device_members(self) -> None:
        self._base = {"local": None, "shared": None}

    @cute.jit
    def ptr(self, name: str) -> cute.Pointer:
        region = self._region_by_name[name]
        base = self._base[region.buffer_space]
        address = base.toint() + Int64(self._offset[name])
        return cute.make_ptr(
            region.dtype, address, AddressSpace.gmem, assumed_align=self._byte_alignment[name]
        )

    @cute.jit
    def tensor(self, name: str) -> cute.Tensor:
        region = self._region_by_name[name]
        return cute.make_tensor(
            self.ptr(name), cute.make_layout(region.shape, stride=self._stride[name])
        )

    @cute.jit
    def reset_tail(self, tid: Int32, total_threads: int) -> None:
        self.reset_tail_space("local", tid, total_threads)
        self.reset_tail_space("shared", tid, total_threads)

    @cute.jit
    def reset_tail_space(self, buffer_space: BufferSpace, tid: Int32, total_threads: int) -> None:
        num_vectors = self._tail_leading[buffer_space] // 16
        if cutlass.const_expr(num_vectors > 0):
            vectors = cute.make_tensor(
                cute.make_ptr(
                    Int32, self._base[buffer_space].toint(), AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout((num_vectors, 4), stride=(4, 1)),
            )
            zero = cute.make_rmem_tensor((4,), Int32)
            for element in cutlass.range_constexpr(4):
                zero[element] = Int32(0)
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Int32, num_bits_per_copy=128
            )
            reset_round_count = ceil_div(num_vectors, total_threads)
            for reset_round in cutlass.range_constexpr(reset_round_count):
                vector_index = Int32(reset_round * total_threads) + tid
                if vector_index < Int32(num_vectors):
                    cute.copy(store_atom, zero, vectors[vector_index, None])

    def _require_finalized(self) -> None:
        if not self._finalized:
            raise RuntimeError("DeviceWorkspace must be finalized first.")
