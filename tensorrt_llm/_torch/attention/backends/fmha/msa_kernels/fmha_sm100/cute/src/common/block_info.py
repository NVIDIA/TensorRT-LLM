# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

from typing import Tuple
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from src.common.seqlen_info import SeqlenInfoQK


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool]
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> Tuple[Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        if const_expr(self.is_causal):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_block_max = min(n_block_max, cute.ceil_div(n_idx, self.tile_n))
        n_block_min = 0
        if num_splits > 1:
            num_n_blocks_per_split = (
                Int32(0)
                if n_block_max <= n_block_min
                else (n_block_max - n_block_min + num_splits - 1) // num_splits
            )
            n_block_min = n_block_min + split_idx * num_n_blocks_per_split
            n_block_max = cutlass.min(n_block_min + num_n_blocks_per_split, n_block_max)
        return n_block_min, n_block_max

    @cute.jit
    def get_m_block_min_max(self, seqlen_info: SeqlenInfoQK, n_block: Int32) -> Tuple[Int32, Int32]:
        m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)
        if const_expr(self.qhead_per_kvhead_packgqa > 1):
            m_block_max = cute.ceil_div(
                seqlen_info.seqlen_q * self.qhead_per_kvhead_packgqa, self.tile_m
            )
        m_block_min = 0
        if const_expr(self.is_causal):
            n_idx_min = n_block * self.tile_n
            m_idx = n_idx_min + seqlen_info.seqlen_q - seqlen_info.seqlen_k
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx *= self.qhead_per_kvhead_packgqa
            m_block_min = cutlass.max(m_block_min, m_idx // self.tile_m)
        return m_block_min, m_block_max
