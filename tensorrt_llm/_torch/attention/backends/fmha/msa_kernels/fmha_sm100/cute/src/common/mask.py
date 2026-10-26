# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

from typing import Callable, Optional, TypeAlias
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr

import src.common.utils as utils
from src.common.seqlen_info import SeqlenInfoQK

MaskGenFn: TypeAlias = Callable[[int], Uint32]
MASK_R2P_CHUNK_SIZE: int = 32


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def r2p_bitmask_above(limit: Int32, s: int) -> Uint32:
    n = max(limit - s * MASK_R2P_CHUNK_SIZE, 0)
    return utils.shl_u32(Uint32(0xFFFFFFFF), Uint32(n))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, MASK_R2P_CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        for i in cutlass.range_constexpr(min(MASK_R2P_CHUNK_SIZE, ncol - s * MASK_R2P_CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * MASK_R2P_CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def row_to_r2p_idx(x: Int32, num_rep: int, num_wg: int) -> Int32:
    return x // (num_rep * num_wg) * num_rep + min(x % (num_rep * num_wg), num_rep)


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        row_idx: Optional[Int32] = None,
        kv_valid_cols: Optional[Int32] = None,
        kv_block_col_start: Optional[Int32] = None,
    ) -> None:
        if const_expr(not mask_seqlen and not mask_causal):
            return

        col_limit = Int32(self.tile_n)
        if const_expr(mask_seqlen):
            if const_expr(kv_valid_cols is not None):
                col_limit = kv_valid_cols
            else:
                col_limit = self.seqlen_k - n_block * Int32(self.tile_n)

        if const_expr(mask_causal):
            if const_expr(row_idx is None):
                row_axis = 0 if const_expr(not self.swap_AB) else 1
                row_idx_cur = tScS_t2r[0][row_axis] + m_block * Int32(self.tile_m)
                if const_expr(self.qhead_per_kvhead_packgqa > 1):
                    row_idx_cur = row_idx_cur // Int32(self.qhead_per_kvhead_packgqa)
            else:
                row_idx_cur = row_idx
            if const_expr(kv_block_col_start is not None):
                block_col_start = kv_block_col_start
            else:
                block_col_start = n_block * Int32(self.tile_n)
            causal_col_limit = (
                row_idx_cur + self.seqlen_k - self.seqlen_q
                - block_col_start + Int32(1)
            )
            col_limit = (
                cutlass.min(col_limit, causal_col_limit)
                if const_expr(mask_seqlen)
                else causal_col_limit
            )

        if col_limit < Int32(self.tile_n):
            mask_r2p_lambda(
                acc_S,
                lambda s: r2p_bitmask_below(col_limit, s),
                rank1=True,
            )

    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        t0ScS_t2r: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
        is_full_block: bool = False,
        check_m_boundary: bool = True,
        valid_tok_count: Optional[Int32] = None,
        q_idx_tile: Optional[cute.Tensor] = None,
        masked_tok_count: Optional[Int32] = None,
    ) -> None:
        del is_full_block, check_m_boundary
        del t0ScS_t2r
        row_axis = 0 if const_expr(not self.swap_AB) else 1
        col_axis = 1 if const_expr(not self.swap_AB) else 0

        if const_expr(valid_tok_count is not None):
            kv_block_col_start = n_block * Int32(self.tile_n)
            causal_q_offset = self.seqlen_k - self.seqlen_q
            nfrag = const_expr(cute.size(acc_S.shape))
            for i in cutlass.range(nfrag, unroll_full=True):
                row_idx = tScS_t2r[i][row_axis]
                tok_idx = row_idx // Int32(self.qhead_per_kvhead_packgqa)
                acc_S[i] = -Float32.inf if tok_idx >= valid_tok_count else acc_S[i]
                if const_expr(mask_seqlen):
                    kv_idx = kv_block_col_start + tScS_t2r[i][col_axis]
                    acc_S[i] = -Float32.inf if kv_idx >= self.seqlen_k else acc_S[i]
                if const_expr(mask_causal):
                    if const_expr(q_idx_tile is not None):
                        causal_tok_count = (
                            masked_tok_count
                            if const_expr(masked_tok_count is not None)
                            else Int32(0)
                        )
                        if tok_idx < causal_tok_count:
                            q_idx = q_idx_tile[tok_idx]
                            kv_idx = kv_block_col_start + tScS_t2r[i][col_axis]
                            acc_S[i] = (
                                -Float32.inf if kv_idx > q_idx + causal_q_offset else acc_S[i]
                            )
            return

        thr_col_offset = tScS_t2r[0][col_axis]
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset

        if const_expr(not mask_causal):
            if const_expr(mask_seqlen) and seqlenk_col_limit <= 0:
                for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                    acc_S[i] = -cutlass.Float32.inf
            return

        thr_row_offset = tScS_t2r[0][row_axis]
        seqlenq_row_limit = self.seqlen_q - m_block * self.tile_m - thr_row_offset
        row_limit_top = seqlenq_row_limit - seqlenk_col_limit
        if const_expr(mask_seqlen) and seqlenk_col_limit <= 0:
            row_limit_top = self.tile_m
        num_rep = cute.size(tScS_t2r, mode=[0])
        row_limit = row_to_r2p_idx(row_limit_top, num_rep, 2)
        mask_r2p_lambda(
            acc_S,
            lambda s: r2p_bitmask_above(row_limit, s),
            rank1=True,
        )
