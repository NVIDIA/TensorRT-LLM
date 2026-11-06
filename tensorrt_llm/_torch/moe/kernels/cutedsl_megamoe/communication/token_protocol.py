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
    """One i64 routing record: ``token:u16 | topk:u8 | rank:u8 | dst_order_id:u8 | (dst_token_cnt-1):u8 | reserved:u16``.
    . ``dst_token_cnt`` = the token's routes landing on the destination rank, ``dst_order_id`` = this
    route's position among them in ascending expert order (``order == cnt - 1`` is the last copy). Producers that do
    not track copies use order 0, count 1."""

    src_rank: Int32
    src_token: Int32
    src_topk: Int32
    dst_order_id: Int32 = Int32(0)
    dst_token_cnt: Int32 = Int32(1)

    nbytes: ClassVar[int] = 8
    FIELD_BITS: ClassVar[int] = 8
    TOKEN_BITS: ClassVar[int] = 16
    MAX_RANKS: ClassVar[int] = 1 << 8
    MAX_TOPK: ClassVar[int] = 1 << 8
    MAX_TOKENS: ClassVar[int] = 1 << 16

    def pack(self) -> Int64:
        high = Int64(self.dst_order_id) | ((Int64(self.dst_token_cnt) - Int64(1)) << Int64(8))
        low = (
            (Int64(self.src_rank) << Int64(24))
            | (Int64(self.src_topk) << Int64(16))
            | (Int64(self.src_token) & Int64(0xFFFF))
        )
        return (high << Int64(32)) | low

    @property
    def is_last_copy(self) -> Int32:
        """1 when this row is the destination rank's last copy of the source token."""
        return Int32(self.dst_order_id == self.dst_token_cnt - Int32(1))

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
    def unpack(cls, packed: Int64) -> "TokenSrcMetadata":
        """Inverse of ``pack`` for a record already in a register; every reader must decode through here."""
        high = packed >> Int64(32)
        return cls(
            src_rank=Int32((packed >> Int64(24)) & Int64(0xFF)),
            src_token=Int32(packed & Int64(0xFFFF)),
            src_topk=Int32((packed >> Int64(16)) & Int64(0xFF)),
            dst_order_id=Int32(high & Int64(0xFF)),
            dst_token_cnt=Int32((high >> Int64(8)) & Int64(0xFF)) + Int32(1),
        )

    @classmethod
    def load(
        cls,
        address: Union[cute.Pointer, Int64],
    ) -> "TokenSrcMetadata":
        return cls.unpack(Int64(cute.arch.load(cls._pointer(address), Int64, scope="gpu")))


__all__ = ["TokenSrcMetadata"]
