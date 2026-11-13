# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Cross-rank token metadata protocol."""

import dataclasses
from typing import ClassVar, Union

import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64

# Relay records of the internode component: the top-k field is
# RELAY_RECORD_TOPK_MARKER | (source slot << RELAY_RECORD_SLOT_SHIFT) | top-k slot of the route
# inside the relay message; src_token is the message index and src_rank the rail peer that landed
# it. The FC2 epilogue decodes the same fields to push the row into the rail peer's landing region.
RELAY_RECORD_TOPK_MARKER = 0x8000
RELAY_RECORD_SLOT_SHIFT = 4
RELAY_RECORD_SLOT_MASK = 0x7FF
RELAY_RECORD_TOPK_MASK = 0xF


@dataclasses.dataclass(frozen=True)
class TokenSrcMetadata:
    """One i64 routing record:
    ``token:u16 | topk_lo:u8 | rank:u8 | dst_order_id:u8 | (dst_token_cnt-1):u8 | topk_hi:u8 | reserved:u8``.

    ``dst_token_cnt`` = the token's routes landing on the destination rank, ``dst_order_id`` = this
    route's position among them in ascending expert order (``order == cnt - 1`` is the last copy). Producers that do
    not track copies use order 0, count 1. ``src_topk`` is carried in 16 bits (low byte at bits 16..23, high byte at
    bits 48..55): plain routes use < MAX_TOPK, the internode component's relay records use
    RELAY_RECORD_TOPK_MARKER | slot << RELAY_RECORD_SLOT_SHIFT | topk, which needs the full 16 bits."""

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
    TOPK_FIELD_BITS: ClassVar[int] = 16   # with wide_topk (internode relay records); 8 otherwise

    def pack(self, *, wide_topk: bool = False) -> Int64:
        """``wide_topk`` carries the 16-bit top-k field of the internode relay records (high byte at
        bits 48..55); the default is main's 8-bit record, which NVLink-domain components keep."""
        if wide_topk:
            topk = Int64(self.src_topk) & Int64(0xFFFF)
            high = (
                Int64(self.dst_order_id)
                | ((Int64(self.dst_token_cnt) - Int64(1)) << Int64(8))
                | ((topk >> Int64(8)) << Int64(16))
            )
            low = (
                (Int64(self.src_rank) << Int64(24))
                | ((topk & Int64(0xFF)) << Int64(16))
                | (Int64(self.src_token) & Int64(0xFFFF))
            )
            return (high << Int64(32)) | low
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

    def store(self, address: Union[cute.Pointer, Int64], *, wide_topk: bool = False) -> None:
        cute.arch.store(self._pointer(address), self.pack(wide_topk=wide_topk), scope="gpu")

    @classmethod
    def unpack(cls, packed: Int64, *, wide_topk: bool = False) -> "TokenSrcMetadata":
        """Inverse of ``pack`` for a record already in a register; every reader must decode through here.
        ``wide_topk`` selects the 16-bit top-k decode of the internode relay records."""
        high = packed >> Int64(32)
        if wide_topk:
            return cls(
                src_rank=Int32((packed >> Int64(24)) & Int64(0xFF)),
                src_token=Int32(packed & Int64(0xFFFF)),
                src_topk=Int32(((packed >> Int64(16)) & Int64(0xFF)) | (((high >> Int64(16)) & Int64(0xFF)) << Int64(8))),
                dst_order_id=Int32(high & Int64(0xFF)),
                dst_token_cnt=Int32((high >> Int64(8)) & Int64(0xFF)) + Int32(1),
            )
        # Main's decode, field order included (keeps the single-node IR identical).
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
        *,
        wide_topk: bool = False,
    ) -> "TokenSrcMetadata":
        return cls.unpack(Int64(cute.arch.load(cls._pointer(address), Int64, scope="gpu")), wide_topk=wide_topk)


__all__ = ["TokenSrcMetadata"]
