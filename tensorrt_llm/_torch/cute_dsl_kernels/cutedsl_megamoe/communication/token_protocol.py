# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Cross-rank token metadata protocol."""

import dataclasses
from typing import ClassVar, Union

import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64

@dataclasses.dataclass(frozen=True)
class TokenSrcMetadata:
    """One i64 routing record: rank:u16, topk:u16, token:u32."""

    src_rank: Int32
    src_token: Int32
    src_topk: Int32

    nbytes: ClassVar[int] = 8

    def pack(self) -> Int64:
        high = (Int64(self.src_rank) << Int64(16)) | Int64(self.src_topk)
        return (high << Int64(32)) | (
            Int64(self.src_token) & Int64(0xFFFFFFFF)
        )

    @staticmethod
    def _pointer(address: Union[cute.Pointer, Int64]) -> cute.Pointer:
        raw_address = (
            address if isinstance(address, Int64) else address.toint()
        )
        return cute.make_ptr(
            Int64,
            raw_address,
            AddressSpace.gmem,
            assumed_align=8,
        )

    def store(self, address: Union[cute.Pointer, Int64]) -> None:
        cute.arch.store(self._pointer(address), self.pack(), scope="gpu")

    @classmethod
    def load(
        cls,
        address: Union[cute.Pointer, Int64],
    ) -> "TokenSrcMetadata":
        packed = Int64(
            cute.arch.load(cls._pointer(address), Int64, scope="gpu")
        )
        high = packed >> Int64(32)
        return cls(
            src_rank=Int32((high >> Int64(16)) & Int64(0xFFFF)),
            src_token=Int32(packed & Int64(0xFFFFFFFF)),
            src_topk=Int32(high & Int64(0xFFFF)),
        )


__all__ = ["TokenSrcMetadata"]
