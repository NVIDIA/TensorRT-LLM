# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Symmetric-heap peer pointer mapper.

The device-side SymBuffer carries ONLY the per-rank offset table.  Layout is
chosen at trace time by ``num_max_ranks`` (a constexpr):

* ``<= 16`` ranks: a ``cute_nvgpu.grid_constant`` / ``llvm.byval`` pointer to a
  ``struct<(array<16 x i64>)>`` (exactly 128B).  ``map`` does a runtime-indexed
  ``getelementptr`` + ``load`` that lowers to a const-bank ``LDC c[bnk][Ra]``.
  The 128B byval size matches the host-side ``createKernelArgs`` copy (hardcoded
  to 128B today), so it is corruption-safe up to EP16.

* ``> 16`` ranks: a by-value ``vector<N x i64>`` read with ``extractelement``.
  Correct but spills the runtime index -- the fallback until the byval-size fix
  reaches public cutedsl.

``base``/``rank_idx`` are intentionally dropped: ``get_base_ptr`` had no callers
and ``rank_idx`` is the owning rank's own constexpr (never read on device), so
freeing those slots lets all 16 byval lanes hold offsets (EP16, not EP14).
"""

from dataclasses import dataclass
from typing import Any, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm
from cutlass.base_dsl.dsl import (extract_mlir_values, get_mlir_types,
                                  new_from_mlir_values)
from cutlass.base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry
from cutlass.base_dsl.typing import get_c_pointers
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op

try:
    # GEP encodes a runtime index by storing this sentinel in rawConstantIndices;
    # it is MLIR's LLVM::GEPOp::kDynamicIndex (== INT32_MIN), frozen by the IR
    # encoding ABI.  Track the canonical constant rather than re-hardcoding it.
    from cutlass.base_dsl.typing import MLIR_DYNAMIC_INDEX
except ImportError:  # older wheels: value is fixed by the GEP encoding ABI
    MLIR_DYNAMIC_INDEX = -(2**31)

_BYVAL_RANK_LIMIT = 16  # struct<(array<16 x i64>)> == exactly 128B


#TODO: Fix this when compiler fixed. This is a shit WAR due to the cuda-to-llvm bug, it just treats any kernel arg to tma_desc as long as it's marked `grid_constant + byval`
def _byval_struct_ty() -> Any:
    """128B byval pointee shared by alloca / GEP / the ``llvm.byval`` attr."""
    return ir.Type.parse(f"!llvm.struct<(array<{_BYVAL_RANK_LIMIT} x i64>)>")


@dataclass(frozen=True)
class SymBufferDeviceBase:
    """Device-side SymBuffer: the per-rank offset table only.

    ``val`` is a ``!llvm.ptr`` to a byval/grid_constant ``struct<(array<16 x i64>)>``
    when ``num_max_ranks <= 16`` (``map`` -> GEP + load -> ``LDC``), else a by-value
    ``vector<N x i64>`` (``map`` -> ``extractelement``, spills).
    """

    val: Any
    num_max_ranks: cutlass.Constexpr[int]

    def __extract_mlir_values__(self) -> list:
        return [self.val]

    def __new_from_mlir_values__(self, values: list) -> "SymBufferDeviceBase":
        return SymBufferDeviceBase(val=values[0],
                                   num_max_ranks=self.num_max_ranks)

    def __get_mlir_types__(self) -> list:
        if self.num_max_ranks <= _BYVAL_RANK_LIMIT:
            return [ir.Type.parse("!llvm.ptr")]
        return [ir.Type.parse(f"vector<{self.num_max_ranks}xi64>")]

    def __extract_mlir_attributes__(self) -> list:
        if self.num_max_ranks <= _BYVAL_RANK_LIMIT:
            return [
                ir.DictAttr.get({
                    "cute_nvgpu.grid_constant":
                    ir.UnitAttr.get(),
                    "llvm.byval":
                    ir.TypeAttr.get(_byval_struct_ty()),
                })
            ]
        return [ir.DictAttr.get({})]

    @cute.jit
    def map(
            self,
            local_ptr: Int64,
            dst_rank_idx: Int32,
            byte_off: Int64 = Int64(0),
    ) -> Int64:
        if cutlass.const_expr(self.num_max_ranks <= _BYVAL_RANK_LIMIT):
            # Opaque ptr -> the offsets array sits at byte 0 of the byval struct,
            # so a flat ``gep i64, ptr, dst_rank`` reaches offsets[dst_rank]
            # directly; the byval struct type only governs the 128B const-bank copy.
            i64_ty = ir.Type.parse("i64")
            off_ptr = llvm.getelementptr(
                ir.Type.parse("!llvm.ptr"),
                self.val,
                [dst_rank_idx.ir_value()],
                [MLIR_DYNAMIC_INDEX],
                i64_ty,
                no_wrap_flags="None",
            )
            off = Int64(llvm.load(i64_ty, off_ptr))
        else:
            off = Int64(llvm.extractelement(self.val, dst_rank_idx.ir_value()))
        return local_ptr + off + byte_off

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

    @dsl_user_op
    def make_device_obj(self, *, loc=None, ip=None) -> Any:
        """Build the offsets-only device obj (see module docstring for layout)."""
        offsets = tuple(self.offsets)
        num_max_ranks = self.num_max_ranks
        if len(offsets) != num_max_ranks:
            raise ValueError(f"len(offsets)={len(offsets)} must equal "
                             f"num_max_ranks={num_max_ranks}.")

        if num_max_ranks <= _BYVAL_RANK_LIMIT:
            ptr_ty = ir.Type.parse("!llvm.ptr")
            st_ty = _byval_struct_ty()
            i64_ty = ir.Type.parse("i64")
            one = arith.constant(
                value=ir.IntegerAttr.get(i64_ty, 1),
                result=i64_ty,
                loc=loc,
                ip=ip,
            )
            buf = llvm.alloca(
                res=ptr_ty,
                elem_type=st_ty,
                array_size=one,
                alignment=64,
                loc=loc,
                ip=ip,
            )
            for i, off in enumerate(offsets):
                slot = llvm.getelementptr(
                    ptr_ty,
                    buf,
                    [],
                    [i],
                    i64_ty,
                    no_wrap_flags="None",
                    loc=loc,
                    ip=ip,
                )
                llvm.store(self._as_int64(off).ir_value(), slot, loc=loc, ip=ip)
            return SymBufferDeviceBase(val=buf, num_max_ranks=num_max_ranks)

        i32_ty = ir.Type.parse("i32")
        vec_ty = ir.Type.parse(f"vector<{num_max_ranks}xi64>")
        vec = llvm.mlir_zero(vec_ty, loc=loc, ip=ip)
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
        return SymBufferDeviceBase(val=vec, num_max_ranks=num_max_ranks)


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
