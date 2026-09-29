# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric-heap peer pointer mapping carried in kernel arguments."""

from dataclasses import dataclass
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm
from cutlass.base_dsl.dsl import extract_mlir_values, get_mlir_types, new_from_mlir_values
from cutlass.base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry
from cutlass.base_dsl.typing import get_c_pointers
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op
from packaging.version import Version

try:
    from cutlass.base_dsl.typing import MLIR_DYNAMIC_INDEX
except ImportError:
    MLIR_DYNAMIC_INDEX = -(2**31)


_dsl_release = Version(Version(cutlass.__version__).base_version)
_grid_constant_width_is_free = _dsl_release < Version("4.0.0") or _dsl_release >= Version("4.7.0")
_byval_rank_limit = 2**31 if _grid_constant_width_is_free else 16


def _byval_struct_type(rank_count: int) -> Any:
    words = rank_count if _grid_constant_width_is_free else _byval_rank_limit
    return ir.Type.parse(f"!llvm.struct<(array<{words} x i64>)>")


@dataclass(frozen=True)
class SymmetricBufferDevice:
    """Device-side peer offset table in constant/by-value kernel arguments."""

    value: Any
    max_ranks: cutlass.Constexpr[int]

    def __extract_mlir_values__(self) -> list:
        return [self.value]

    def __new_from_mlir_values__(self, values: list) -> "SymmetricBufferDevice":
        return SymmetricBufferDevice(values[0], self.max_ranks)

    def __get_mlir_types__(self) -> list:
        if self.max_ranks <= _byval_rank_limit:
            return [ir.Type.parse("!llvm.ptr")]
        return [ir.Type.parse(f"vector<{self.max_ranks}xi64>")]

    def __extract_mlir_attributes__(self) -> list:
        if self.max_ranks <= _byval_rank_limit:
            return [
                ir.DictAttr.get(
                    {
                        "cute_nvgpu.grid_constant": ir.UnitAttr.get(),
                        "llvm.byval": ir.TypeAttr.get(_byval_struct_type(self.max_ranks)),
                    }
                )
            ]
        return [ir.DictAttr.get({})]

    @cute.jit
    def map(
        self, local_address: Int64, destination_rank: Int32, byte_offset: Int64 = Int64(0)
    ) -> Int64:
        if cutlass.const_expr(self.max_ranks <= _byval_rank_limit):
            i64_type = ir.Type.parse("i64")
            offset_pointer = llvm.getelementptr(
                ir.Type.parse("!llvm.ptr"),
                self.value,
                [destination_rank.ir_value()],
                [MLIR_DYNAMIC_INDEX],
                i64_type,
                no_wrap_flags="None",
            )
            peer_offset = Int64(llvm.load(i64_type, offset_pointer))
        else:
            peer_offset = Int64(llvm.extractelement(self.value, destination_rank.ir_value()))
        return local_address + peer_offset + byte_offset

    @cute.jit
    def map_pointer(self, pointer, destination_rank: Int32, byte_alignment: Optional[int] = None):
        if cutlass.const_expr(pointer.memspace != AddressSpace.gmem):
            raise ValueError("Only GMEM pointers can be mapped to a symmetric peer.")
        if cutlass.const_expr(byte_alignment is None):
            byte_alignment = pointer.max_alignment
        return cute.make_ptr(
            pointer.dtype,
            self.map(pointer.toint(), destination_rank),
            pointer.memspace,
            assumed_align=byte_alignment,
        )


@dataclass(frozen=True)
class SymmetricBufferHost:
    """Host launch payload used to construct a SymmetricBufferDevice."""

    base_address: Int64
    offsets: tuple
    rank: Int32
    max_ranks: cutlass.Constexpr[int]

    @staticmethod
    def _as_int64(value) -> Int64:
        return value if isinstance(value, Int64) else Int64(int(value))

    @dsl_user_op
    def make_device_object(self, *, loc=None, ip=None) -> SymmetricBufferDevice:
        offsets = tuple(self.offsets)
        if len(offsets) != self.max_ranks:
            raise ValueError(f"Expected {self.max_ranks} peer offsets, got {len(offsets)}.")

        if self.max_ranks <= _byval_rank_limit:
            pointer_type = ir.Type.parse("!llvm.ptr")
            struct_type = _byval_struct_type(self.max_ranks)
            i64_type = ir.Type.parse("i64")
            one = arith.constant(
                value=ir.IntegerAttr.get(i64_type, 1), result=i64_type, loc=loc, ip=ip
            )
            buffer = llvm.alloca(
                res=pointer_type,
                elem_type=struct_type,
                array_size=one,
                alignment=64,
                loc=loc,
                ip=ip,
            )
            for index, offset in enumerate(offsets):
                slot = llvm.getelementptr(
                    pointer_type,
                    buffer,
                    [],
                    [index],
                    i64_type,
                    no_wrap_flags="None",
                    loc=loc,
                    ip=ip,
                )
                llvm.store(self._as_int64(offset).ir_value(), slot, loc=loc, ip=ip)
            return SymmetricBufferDevice(buffer, self.max_ranks)

        i32_type = ir.Type.parse("i32")
        vector_type = ir.Type.parse(f"vector<{self.max_ranks}xi64>")
        vector = llvm.mlir_zero(vector_type, loc=loc, ip=ip)
        for index, offset in enumerate(offsets):
            element_index = arith.constant(
                value=ir.IntegerAttr.get(i32_type, index), result=i32_type, loc=loc, ip=ip
            )
            vector = llvm.insertelement(
                vector, self._as_int64(offset).ir_value(), element_index, loc=loc, ip=ip
            )
        return SymmetricBufferDevice(vector, self.max_ranks)


@JitArgAdapterRegistry.register_jit_arg_adapter(SymmetricBufferHost)
class _SymmetricBufferHostAdapter:
    def __init__(self, argument: SymmetricBufferHost) -> None:
        self._argument = argument
        offsets = tuple(argument.offsets)
        if len(offsets) != int(argument.max_ranks):
            raise ValueError(
                f"Expected {int(argument.max_ranks)} peer offsets, got {len(offsets)}."
            )
        self._fields = (
            Int64(argument.base_address),
            *(Int64(offset) for offset in offsets),
            Int32(argument.rank),
        )

    def __c_pointers__(self) -> list[Any]:
        pointers: list[Any] = []
        for field in self._fields:
            pointers.extend(get_c_pointers(field))
        return pointers

    def __get_mlir_types__(self) -> list[Any]:
        types: list[Any] = []
        for field in self._fields:
            types.extend(get_mlir_types(field))
        return types

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values: list[ir.Value] = []
        for field in self._fields:
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> SymmetricBufferHost:
        value_index = 0
        rebuilt = []
        for field in self._fields:
            field_value_count = len(get_mlir_types(field))
            rebuilt.append(
                new_from_mlir_values(field, values[value_index : value_index + field_value_count])
            )
            value_index += field_value_count
        if value_index != len(values):
            raise ValueError(f"Consumed {value_index} MLIR values, got {len(values)}.")

        result = object.__new__(SymmetricBufferHost)
        object.__setattr__(result, "base_address", rebuilt[0])
        object.__setattr__(result, "offsets", tuple(rebuilt[1:-1]))
        object.__setattr__(result, "rank", rebuilt[-1])
        object.__setattr__(result, "max_ranks", self._argument.max_ranks)
        return result


__all__ = ["SymmetricBufferDevice", "SymmetricBufferHost"]
