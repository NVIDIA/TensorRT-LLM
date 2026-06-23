# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Symmetric-heap peer pointer mapper.

``SymBufferHost`` is the runtime payload that crosses the Python ->
generated-host-code boundary.  Inside the generated host wrapper, it packs
the runtime base address and per-rank offsets into a device-side
``SymBuffer{N}`` native struct:

    { i64 base, vector<N x i64> offsets, i32 rank_idx }

Device code only sees that struct and calls ``.map`` /
``.ptr_map_to_rank``.  The vector field is deliberate: LLVM supports
runtime-indexed ``extractelement`` on vectors, which NVPTX lowers to an
indexed param-bank load (``LDC.U64``).
"""

from dataclasses import dataclass
from typing import Any, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm
from cutlass.base_dsl.dsl import (extract_mlir_values, get_mlir_types,
                                  new_from_mlir_values)
from cutlass.base_dsl.native_struct import native_struct
from cutlass.base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry
from cutlass.base_dsl.typing import get_c_pointers
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op


@dataclass(frozen=True)
class SymBufferDeviceBase:
    """Device-side methods shared by all generated ``SymBuffer{N}`` types."""

    @cute.jit
    def map(
            self,
            local_ptr: Int64,
            dst_rank_idx: Int32,
            byte_off: Int64 = Int64(0),
    ) -> Int64:
        off = Int64(llvm.extractelement(self.offsets, dst_rank_idx.ir_value()))
        return local_ptr + off + byte_off

    @cute.jit
    def get_base_ptr(self) -> Int64:
        return self.base

    @cute.jit
    def ptr_map_to_rank(self, ptr, dst_rank_idx: Int32):
        if cutlass.const_expr(ptr.memspace != AddressSpace.gmem):
            raise ValueError(
                f"ptr_map_to_rank: source pointer must live in GMEM "
                f"(NVSHMEM symmetric heap), got memspace={ptr.memspace}.")
        peer_addr = self.map(ptr.toint(), dst_rank_idx, Int64(0))
        return cute.make_ptr(
            ptr.dtype,
            peer_addr,
            ptr.memspace,
            assumed_align=ptr.max_alignment,
        )


@dataclass(frozen=True)
class SymBufferHost:
    """Runtime launch payload for a device-side ``SymBuffer{N}``."""

    base_addr: int
    offsets: Tuple[int, ...]
    rank_idx: int
    num_max_ranks: cutlass.Constexpr[int]

    @staticmethod
    def _as_int64(value) -> Int64:
        return value if isinstance(value, Int64) else Int64(int(value))

    @staticmethod
    def _as_int32(value) -> Int32:
        return value if isinstance(value, Int32) else Int32(int(value))

    @staticmethod
    def _make_device_type(num_max_ranks: int) -> type:
        if num_max_ranks <= 0:
            raise ValueError(
                f"num_max_ranks must be positive, got {num_max_ranks}")

        vec_ty_str = f"vector<{num_max_ranks}xi64>"

        class _OffsetsT:

            @staticmethod
            def mlir_type() -> ir.Type:
                return ir.Type.parse(vec_ty_str)

        @native_struct
        class _SymBufferDevice(SymBufferDeviceBase):
            base: Int64
            offsets: _OffsetsT
            rank_idx: Int32

        cls = _SymBufferDevice
        cls.__name__ = f"SymBuffer{num_max_ranks}"
        cls.__qualname__ = cls.__name__
        cls.NUM_MAX_RANKS = num_max_ranks
        return cls

    @dsl_user_op
    def make_device_obj(self, *, loc=None, ip=None) -> Any:
        offsets = tuple(self.offsets)
        num_max_ranks = self.num_max_ranks
        if len(offsets) != num_max_ranks:
            raise ValueError(
                f"len(offsets)={len(offsets)} must equal "
                f"num_max_ranks={num_max_ranks}; SymBuffer requires its "
                f"runtime payload length to match the compiled vector type.")

        vec_ty = ir.Type.parse(f"vector<{num_max_ranks}xi64>")
        vec = llvm.mlir_zero(vec_ty, loc=loc, ip=ip)
        i32_ty = ir.Type.parse("i32")
        for i, off in enumerate(offsets):
            idx = arith.constant(
                value=ir.IntegerAttr.get(i32_ty, i),
                result=i32_ty,
                loc=loc,
                ip=ip,
            )
            vec = llvm.insertelement(
                vec,
                self._as_int64(off).ir_value(),
                idx,
                loc=loc,
                ip=ip,
            )

        return self._make_device_type(num_max_ranks)(
            base=self._as_int64(self.base_addr),
            offsets=vec,
            rank_idx=self._as_int32(self.rank_idx),
            loc=loc,
            ip=ip,
        )


@JitArgAdapterRegistry.register_jit_arg_adapter(SymBufferHost)
class _SymBufferHostAdapter:
    """JIT boundary adapter for ``SymBufferHost``.

    Python-side ``SymBufferHost`` stays pure host data (ints + tuple).  The
    adapter is the only place that maps it to DSL scalar arguments:
    base/offsets are i64, rank_idx is i32, and num_max_ranks remains a
    constexpr carried through reconstruction.
    """

    def __init__(self, arg: SymBufferHost) -> None:
        self._arg = arg
        if len(tuple(arg.offsets)) != int(arg.num_max_ranks):
            raise ValueError(
                f"len(offsets)={len(tuple(arg.offsets))} must equal "
                f"num_max_ranks={int(arg.num_max_ranks)}.")
        self._fields = (
            Int64(arg.base_addr),
            *(Int64(x) for x in arg.offsets),
            Int32(arg.rank_idx),
        )

    def __c_pointers__(self) -> list[Any]:
        c_pointers: list[Any] = []
        for field in self._fields:
            c_pointers.extend(get_c_pointers(field))
        return c_pointers

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

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> SymBufferHost:
        idx = 0

        base_n = len(get_mlir_types(self._fields[0]))
        base_addr = new_from_mlir_values(self._fields[0],
                                         values[idx:idx + base_n])
        idx += base_n

        offsets = []
        for field in self._fields[1:-1]:
            n = len(get_mlir_types(field))
            offsets.append(new_from_mlir_values(field, values[idx:idx + n]))
            idx += n

        rank_n = len(get_mlir_types(self._fields[-1]))
        rank_idx = new_from_mlir_values(self._fields[-1],
                                        values[idx:idx + rank_n])
        idx += rank_n
        if idx != len(values):
            raise ValueError(
                f"SymBufferHost adapter consumed {idx} values, got {len(values)}"
            )

        obj = object.__new__(SymBufferHost)
        object.__setattr__(obj, "base_addr", base_addr)
        object.__setattr__(obj, "offsets", tuple(offsets))
        object.__setattr__(obj, "rank_idx", rank_idx)
        object.__setattr__(obj, "num_max_ranks", self._arg.num_max_ranks)
        return obj
