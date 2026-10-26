# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr


@dataclass(frozen=True)
class PagedKVManager:
    mPageTable: cute.Tensor
    page_size: cutlass.Constexpr[int]
    n_block_size: cutlass.Constexpr[int]

    @staticmethod
    def create(
        mPageTable: cute.Tensor,
        *,
        page_size: int,
        n_block_size: int,
    ):
        if page_size != n_block_size:
            raise ValueError(
                f"page_size ({page_size}) must equal blk_kv ({n_block_size})"
            )
        return PagedKVManager(
            mPageTable,
            page_size=page_size,
            n_block_size=n_block_size,
        )

    @cute.jit
    def logical_length(
        self,
        batch_idx: Int32,
        num_kv_blocks: Int32,
        mSeqUsedK=None,
    ) -> Int32:
        if const_expr(mSeqUsedK is not None):
            return mSeqUsedK[batch_idx]
        return num_kv_blocks * Int32(self.n_block_size)

    @cute.jit
    def valid_cols_in_block(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
        num_kv_blocks: Int32,
        mSeqUsedK=None,
    ) -> Int32:
        seqlen_k = self.logical_length(batch_idx, num_kv_blocks, mSeqUsedK)
        block_start = kv_block_idx * Int32(self.n_block_size)
        remaining = seqlen_k - block_start
        remaining = cutlass.max(remaining, Int32(0))
        return cutlass.min(remaining, Int32(self.n_block_size))

    @cute.jit
    def physical_block_index(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
    ) -> Int32:
        return self.mPageTable[batch_idx, kv_block_idx]

__all__ = ["PagedKVManager"]
