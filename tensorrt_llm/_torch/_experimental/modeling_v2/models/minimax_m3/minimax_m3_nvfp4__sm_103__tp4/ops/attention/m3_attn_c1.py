# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The c1 kernel: fused MSA decode attention + o_proj GEMV in one 96-CTA launch, for a one-request, 6-token
verify layer."""

from __future__ import annotations

from typing import NamedTuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .m3_attn_mega import HIDDEN, check_handoff_at_rest, workspace_parity
from .m3_fused_decode_attn import (
    HEAD_DIM,
    MXFP8_BLOCK,
    NUM_HEADS,
    check_attention_inputs,
    mx_gemm,
    mxfp8_rows,
    reference_attention,
    unswizzle_scales,
)

#: 6 clusters of 16 CTAs, each an attention CTA and an o_proj unit: ``flags[32 + CTA]`` are the call counters.
GRID = 96


class C1State(NamedTuple):
    """The c1 kernel's results: the attention's bf16 output, the caches after the insert, and the o_proj output."""

    output: torch.Tensor
    kv_data: torch.Tensor
    kv_scale: torch.Tensor
    idx_k: torch.Tensor
    y: torch.Tensor


class M3AttnC1(OpWrapper):
    """The attention of a one-request verify step of 6 query tokens and its o_proj, in one 96-CTA launch.

    The fused attention of ``m3_fused_decode_attn`` on the packed QKV rows (bf16 ``output``, MXFP8 hand-off into the
    call's parity of ``mxfp8_data`` / ``mxfp8_scale``), and the MXFP8 o_proj GEMV into ``y``: every attention CTA
    is also an o_proj unit (48 row tiles x 2 K splits). ``flags[32 + CTA]`` are the per-CTA call counters; the
    workspace is allocated once and never reset. With ``ar_mode`` 2, ``y`` is also pushed into the next multicast
    all-reduce's exchange at ``ar_workspace`` / ``ar_flags``.

    Returns None; the cells read ``output``, ``y``, the hand-off and the caches.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Eagle3 verify of 5 drafts (qlen 6) a few blocks in; both call parities (two calls, one workspace)",
            spec=dict(seq_len=300, calls=2),
        ),
        Cell(
            why="the full 16-block window: the six tokens fill the last rows of the last page",
            spec=dict(seq_len=2048, calls=1),
        ),
        Cell(
            why="the six tokens straddle a page boundary (positions 126..131)",
            spec=dict(seq_len=132, calls=1),
        ),
        Cell(why="the first six tokens of a request", spec=dict(seq_len=6, calls=1)),
    )

    note = """
    Compiled for the one shape the target launches: one request of 6 query tokens (geometry (6, 1, 1), 96 CTAs),
    the MXFP8 hand-off. The all-reduce push (`ar_mode` 2) needs a multicast workspace and is not driven here: the
    cells run `ar_mode` 0, where `y` is the o_proj output and nothing else is written.
    """

    def __call__(
        self,
        packed: torch.Tensor,
        flags: torch.Tensor,
        kv_data: torch.Tensor,
        kv_scale: torch.Tensor,
        idx_k: torch.Tensor,
        out_cache_loc: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_quant_scale: torch.Tensor,
        k_global_scale: torch.Tensor,
        v_global_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        index_k_norm_weight: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        eps: float,
        sm_scale: float,
        decode_query_len: int,
        tokens_per_cluster: int,
        blocks_per_cta: int,
        output: torch.Tensor,
        mxfp8_data: torch.Tensor,
        mxfp8_scale: torch.Tensor,
        use_pdl: bool,
        o_weight: torch.Tensor,
        o_weight_scale: torch.Tensor,
        y: torch.Tensor,
        ar_workspace: int,
        ar_flags: int,
        ar_rank: int,
        ar_world: int,
        ar_max_tokens: int,
        ar_mode: int,
    ) -> None:
        torch.ops.trtllm.minimax_m3_attn_c1(
            packed,
            flags,
            kv_data,
            kv_scale,
            idx_k,
            out_cache_loc,
            position_ids,
            block_table,
            seq_lens,
            kv_quant_scale,
            k_global_scale,
            v_global_scale,
            q_norm_weight,
            k_norm_weight,
            index_k_norm_weight,
            rotary_cos_sin,
            eps,
            sm_scale,
            decode_query_len,
            tokens_per_cluster,
            blocks_per_cta,
            output,
            mxfp8_data,
            mxfp8_scale,
            use_pdl,
            o_weight,
            o_weight_scale,
            y,
            ar_workspace,
            ar_flags,
            ar_rank,
            ar_world,
            ar_max_tokens,
            ar_mode,
        )

    def reference(
        self,
        packed: torch.Tensor,
        flags: torch.Tensor,
        kv_data: torch.Tensor,
        kv_scale: torch.Tensor,
        idx_k: torch.Tensor,
        out_cache_loc: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_quant_scale: torch.Tensor,
        k_global_scale: torch.Tensor,
        v_global_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        index_k_norm_weight: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        eps: float,
        sm_scale: float,
        decode_query_len: int,
        tokens_per_cluster: int,
        blocks_per_cta: int,
        output: torch.Tensor,
        mxfp8_data: torch.Tensor,
        mxfp8_scale: torch.Tensor,
        use_pdl: bool,
        o_weight: torch.Tensor,
        o_weight_scale: torch.Tensor,
        y: torch.Tensor,
        ar_workspace: int,
        ar_flags: int,
        ar_rank: int,
        ar_world: int,
        ar_max_tokens: int,
        ar_mode: int,
    ) -> C1State:
        """The composition: the fused attention's reference on ``packed``, then
        ``y = bf16(mxfp8(output) @ dequant(W_o)^T)`` (fp64 accumulation, one rounding).

        The workspace, the output buffers, the geometry, ``use_pdl`` and the all-reduce arguments are ignored.
        """
        attn = reference_attention(
            packed,
            kv_data,
            kv_scale,
            idx_k,
            out_cache_loc,
            position_ids,
            block_table,
            seq_lens,
            kv_quant_scale,
            k_global_scale,
            v_global_scale,
            q_norm_weight,
            k_norm_weight,
            index_k_norm_weight,
            rotary_cos_sin,
            eps,
            sm_scale,
            decode_query_len,
        )
        q8, ex = mxfp8_rows(attn.output)
        width = NUM_HEADS * HEAD_DIM
        out_y = mx_gemm(
            q8,
            ex,
            o_weight.view(torch.uint8),
            unswizzle_scales(o_weight_scale, HIDDEN, width // MXFP8_BLOCK),
        )
        return C1State(attn.output, attn.kv_data, attn.kv_scale, attn.idx_k, out_y)

    def is_valid(
        self,
        packed: torch.Tensor,
        flags: torch.Tensor,
        kv_data: torch.Tensor,
        kv_scale: torch.Tensor,
        idx_k: torch.Tensor,
        out_cache_loc: torch.Tensor,
        position_ids: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_quant_scale: torch.Tensor,
        k_global_scale: torch.Tensor,
        v_global_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        index_k_norm_weight: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        eps: float,
        sm_scale: float,
        decode_query_len: int,
        tokens_per_cluster: int,
        blocks_per_cta: int,
        output: torch.Tensor,
        mxfp8_data: torch.Tensor,
        mxfp8_scale: torch.Tensor,
        use_pdl: bool,
        o_weight: torch.Tensor,
        o_weight_scale: torch.Tensor,
        y: torch.Tensor,
        ar_workspace: int,
        ar_flags: int,
        ar_rank: int,
        ar_world: int,
        ar_max_tokens: int,
        ar_mode: int,
    ) -> None:
        check_attention_inputs(
            packed.shape[0],
            kv_data,
            kv_scale,
            idx_k,
            out_cache_loc,
            position_ids,
            block_table,
            seq_lens,
            q_norm_weight,
            k_norm_weight,
            index_k_norm_weight,
            rotary_cos_sin,
            decode_query_len,
        )
        check_handoff_at_rest(mxfp8_data, mxfp8_scale, workspace_parity(flags, GRID))
        assert ar_mode != 2 or (ar_workspace != 0 and ar_flags != 0), (
            "ar_mode 2 needs the exchange's addresses"
        )


m3_attn_c1 = M3AttnC1()
