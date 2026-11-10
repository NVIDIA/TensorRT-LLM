"""Node-local peer pointer mapping over an NCCL LSA-registered arena.

Drop-in replacement for :class:`SymmetricBufferHost` / :class:`SymmetricBufferDevice`
(``nvlink_domain/symmetric_buffer.py``) when the symmetric workspace lives in an NCCL
LSA window instead of the NVSHMEM heap. The device object carries one address delta per
rank of the caller's NVLink domain; kernel call sites keep passing global ranks, ``map``
converts them to the node-local index and executes a device ``trap`` for a cross-node
rank, so an absent LSA mapping can never silently alias the local allocation. Ported from
the legacy ``pre_unification/comm/lsa_peer_mapper.py`` (the standalone MegaMoE implementation outside ``next/``,
the non-next/ code path) with the next/ method names.
"""

from __future__ import annotations

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

from ..nvlink_domain.symmetric_buffer import (
    MLIR_DYNAMIC_INDEX,
    SymmetricBufferDevice,
    _byval_rank_limit,
    _byval_struct_type,
)


@dsl_user_op
def _device_trap(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], "trap;", "", has_side_effects=True, asm_dialect=0, loc=loc, ip=ip)


@dataclass(frozen=True)
class LsaPeerMapperDevice(SymmetricBufferDevice):
    """Device-side node-local offset table plus the caller's global rank.

    A :class:`SymmetricBufferDevice` (the kernels annotate their mapper arguments with that
    type) whose ``max_ranks`` is the NVLink-domain size and whose ``map`` converts the global
    destination rank to the node-local table index.
    """

    rank: Int32 = None  # type: ignore[assignment]  # dataclass inheritance needs a default after the base fields

    @property
    def ranks_per_node(self) -> int:
        return int(self.max_ranks)

    def __extract_mlir_values__(self) -> list:
        return [self.value, self.rank.ir_value()]

    def __new_from_mlir_values__(self, values: list) -> "LsaPeerMapperDevice":
        return LsaPeerMapperDevice(values[0], self.max_ranks, Int32(values[1]))

    def __get_mlir_types__(self) -> list:
        if self.ranks_per_node <= _byval_rank_limit:
            offsets_type = ir.Type.parse("!llvm.ptr")
        else:
            offsets_type = ir.Type.parse(f"vector<{self.ranks_per_node}xi64>")
        return [offsets_type, ir.Type.parse("i32")]

    def __extract_mlir_attributes__(self) -> list:
        if self.ranks_per_node <= _byval_rank_limit:
            offsets_attributes = ir.DictAttr.get(
                {
                    "cute_nvgpu.grid_constant": ir.UnitAttr.get(),
                    "llvm.byval": ir.TypeAttr.get(_byval_struct_type(self.ranks_per_node)),
                }
            )
        else:
            offsets_attributes = ir.DictAttr.get({})
        return [offsets_attributes, ir.DictAttr.get({})]

    @cute.jit
    def _offset_for_local_peer(self, peer_local_index: Int32) -> Int64:
        if cutlass.const_expr(self.ranks_per_node <= _byval_rank_limit):
            i64_type = ir.Type.parse("i64")
            offset_pointer = llvm.getelementptr(
                ir.Type.parse("!llvm.ptr"),
                self.value,
                [peer_local_index.ir_value()],
                [MLIR_DYNAMIC_INDEX],
                i64_type,
                no_wrap_flags="None",
            )
            return Int64(llvm.load(i64_type, offset_pointer))
        return Int64(llvm.extractelement(self.value, peer_local_index.ir_value()))

    @cute.jit
    def map(self, local_address: Int64, destination_rank: Int32, byte_offset: Int64 = Int64(0)) -> Int64:
        """Map a local arena address to the same-node global ``destination_rank``; trap otherwise."""
        if destination_rank < Int32(0):
            _device_trap()
        my_node = self.rank // Int32(self.ranks_per_node)
        destination_node = destination_rank // Int32(self.ranks_per_node)
        if destination_node != my_node:
            _device_trap()
        peer_local_index = destination_rank % Int32(self.ranks_per_node)
        return local_address + self._offset_for_local_peer(peer_local_index) + byte_offset

    @cute.jit
    def map_pointer(self, pointer, destination_rank: Int32, byte_alignment: Optional[int] = None):
        if cutlass.const_expr(pointer.memspace != AddressSpace.gmem):
            raise ValueError("Only GMEM pointers can be mapped to an LSA peer.")
        if cutlass.const_expr(byte_alignment is None):
            byte_alignment = pointer.max_alignment
        return cute.make_ptr(
            pointer.dtype, self.map(pointer.toint(), destination_rank), pointer.memspace, assumed_align=byte_alignment
        )


@dataclass(frozen=True)
class LsaPeerMapperHost:
    """Host launch payload: node-local peer deltas (``offsets[i] = peer_i_base - local_base``)."""

    base_address: Int64
    offsets: tuple
    rank: Int32
    ranks_per_node: cutlass.Constexpr[int]

    def validate(self) -> None:
        """Check the concrete payload (not ``__post_init__``: CuTe rebuilds the dataclass with symbolic scalars)."""
        ranks_per_node = int(self.ranks_per_node)
        rank = int(self.rank)
        if ranks_per_node <= 0:
            raise ValueError(f"ranks_per_node must be positive, got {ranks_per_node}.")
        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}.")
        if len(tuple(self.offsets)) != ranks_per_node:
            raise ValueError(f"len(offsets)={len(tuple(self.offsets))} must equal ranks_per_node={ranks_per_node}.")
        # No zero check on the rank's own entry: the LSA window exposes a rank's own slot through
        # an alias address (a non-zero self offset was observed on every rank), which is valid
        # to dereference and is what the legacy runner has always passed through.

    def map_peer_offset(self, destination_rank: int) -> int:
        """Host-side same-node delta; rejects cross-node ranks like the device ``map``."""
        peer = int(destination_rank)
        rank = int(self.rank)
        ranks_per_node = int(self.ranks_per_node)
        if peer < 0 or peer // ranks_per_node != rank // ranks_per_node:
            raise ValueError(f"cross-node LSA mapping rejected: rank={rank} peer={peer} ranks_per_node={ranks_per_node}.")
        return int(tuple(self.offsets)[peer % ranks_per_node])

    @staticmethod
    def _as_int64(value) -> Int64:
        return value if isinstance(value, Int64) else Int64(int(value))

    @staticmethod
    def _as_int32(value) -> Int32:
        return value if isinstance(value, Int32) else Int32(int(value))

    @dsl_user_op
    def make_device_object(self, *, loc=None, ip=None) -> LsaPeerMapperDevice:
        offsets = tuple(self.offsets)
        ranks_per_node = int(self.ranks_per_node)
        if len(offsets) != ranks_per_node:
            raise ValueError(f"Expected {ranks_per_node} peer offsets, got {len(offsets)}.")
        if ranks_per_node <= _byval_rank_limit:
            pointer_type = ir.Type.parse("!llvm.ptr")
            struct_type = _byval_struct_type(ranks_per_node)
            i64_type = ir.Type.parse("i64")
            one = arith.constant(value=ir.IntegerAttr.get(i64_type, 1), result=i64_type, loc=loc, ip=ip)
            buffer = llvm.alloca(res=pointer_type, elem_type=struct_type, array_size=one, alignment=64, loc=loc, ip=ip)
            for index, offset in enumerate(offsets):
                slot = llvm.getelementptr(
                    pointer_type, buffer, [], [index], i64_type, no_wrap_flags="None", loc=loc, ip=ip
                )
                llvm.store(self._as_int64(offset).ir_value(), slot, loc=loc, ip=ip)
            value = buffer
        else:
            i32_type = ir.Type.parse("i32")
            vector_type = ir.Type.parse(f"vector<{ranks_per_node}xi64>")
            value = llvm.mlir_zero(vector_type, loc=loc, ip=ip)
            for index, offset in enumerate(offsets):
                element_index = arith.constant(
                    value=ir.IntegerAttr.get(i32_type, index), result=i32_type, loc=loc, ip=ip
                )
                value = llvm.insertelement(value, self._as_int64(offset).ir_value(), element_index, loc=loc, ip=ip)
        return LsaPeerMapperDevice(value, ranks_per_node, self._as_int32(self.rank))


@JitArgAdapterRegistry.register_jit_arg_adapter(LsaPeerMapperHost)
class _LsaPeerMapperHostAdapter:
    def __init__(self, argument: LsaPeerMapperHost) -> None:
        self._argument = argument
        offsets = tuple(argument.offsets)
        if len(offsets) != int(argument.ranks_per_node):
            raise ValueError(f"Expected {int(argument.ranks_per_node)} peer offsets, got {len(offsets)}.")
        self._fields = (Int64(argument.base_address), *(Int64(offset) for offset in offsets), Int32(argument.rank))

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

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> LsaPeerMapperHost:
        value_index = 0
        rebuilt = []
        for field in self._fields:
            field_value_count = len(get_mlir_types(field))
            rebuilt.append(new_from_mlir_values(field, values[value_index : value_index + field_value_count]))
            value_index += field_value_count
        if value_index != len(values):
            raise ValueError(f"Consumed {value_index} MLIR values, got {len(values)}.")
        mapper = object.__new__(LsaPeerMapperHost)
        object.__setattr__(mapper, "base_address", rebuilt[0])
        object.__setattr__(mapper, "offsets", tuple(rebuilt[1:-1]))
        object.__setattr__(mapper, "rank", rebuilt[-1])
        object.__setattr__(mapper, "ranks_per_node", self._argument.ranks_per_node)
        return mapper


__all__ = ["LsaPeerMapperDevice", "LsaPeerMapperHost"]
