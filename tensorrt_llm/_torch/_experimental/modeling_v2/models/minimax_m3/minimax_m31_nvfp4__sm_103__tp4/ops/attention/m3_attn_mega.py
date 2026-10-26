# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention megakernel: QKV GEMV + fused MSA decode attention + o_proj GEMV in one launch, for a one-request,
one-token decode layer."""

from __future__ import annotations

from typing import NamedTuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .m3_fused_decode_attn import (
    HEAD_DIM,
    MXFP8_BLOCK,
    NUM_HEADS,
    PACKED_WIDTH,
    check_attention_inputs,
    mx_gemm,
    mxfp8_rows,
    reference_attention,
    unswizzle_scales,
)

#: The layer input (hidden size) and the o_proj output width at tp4.
HIDDEN = 6144
#: Persistent grid of the megakernel (7 clusters of 16): ``flags[32 + CTA]`` are its per-CTA call counters.
GRID = 112
#: Rows of each call-parity buffer.
MAX_TOKENS = 8


class MegaState(NamedTuple):
    """The megakernel's results: the packed QKV rows of the call, the attention's bf16 output, the caches after the
    insert, and the o_proj output."""

    packed: torch.Tensor
    output: torch.Tensor
    kv_data: torch.Tensor
    kv_scale: torch.Tensor
    idx_k: torch.Tensor
    y: torch.Tensor


def workspace_parity(flags: torch.Tensor, grid: int) -> int:
    """The call parity every CTA of the grid will use (their counters move together), checked."""
    counters = flags[32 : 32 + grid]
    assert bool((counters == counters[0]).all()), (
        "the per-CTA call counters disagree: the workspace is corrupt"
    )
    return int(counters[0]) & 1


def check_handoff_at_rest(mxfp8_data: torch.Tensor, mxfp8_scale: torch.Tensor, parity: int) -> None:
    """The MXFP8 hand-off's parity buffer this call writes must hold the 0xFF "not yet" marker (Lamport)."""
    assert bool((mxfp8_data.view(torch.uint8).reshape(2, -1)[parity] == 0xFF).all()) and bool(
        (mxfp8_scale.view(torch.uint8).reshape(2, -1)[parity] == 0xFF).all()
    ), (
        "the MXFP8 hand-off's buffer for this call is not at rest (0xFF): the o_proj units would read stale codes"
    )


class M3AttnMega(OpWrapper):
    """The whole attention half of a decode layer for one request and one query token, in one launch.

    The MXFP8 QKV GEMV of the layer input (as the preceding all-reduce hands it over: E4M3 rows + 128x4-swizzled
    UE8M0 scales) into the call's parity buffer of ``qkv_out``, the fused attention of ``m3_fused_decode_attn`` on
    those rows (bf16 ``output``, MXFP8 hand-off into the call's parity of ``mxfp8_data`` / ``mxfp8_scale``), and the
    MXFP8 o_proj GEMV into ``y``. ``flags`` holds the per-tile ready counters and the per-CTA call counters; the
    workspace is allocated once and never reset. With ``ar_mode`` 2, ``y`` is also pushed into the next multicast
    all-reduce's exchange at ``ar_workspace`` / ``ar_flags``.

    Returns None; the cells read ``qkv_out``, ``output``, ``y``, the hand-off and the caches.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode of one request a few blocks in; both call parities (two calls on one workspace)",
            spec=dict(seq_len=300, calls=2),
        ),
        Cell(
            why="the full 16-block window, the new token on the last row of the last page",
            spec=dict(seq_len=2048, calls=1),
        ),
        Cell(
            why="the new token opens a page (position 1024): its own page's other rows are past the length",
            spec=dict(seq_len=1025, calls=1),
        ),
        Cell(why="the first token of a request: one attended row", spec=dict(seq_len=1, calls=1)),
    )

    note = """
    Compiled for the one shape the target launches: one request, one query token (geometry (1, 1, 1)), the MXFP8
    hand-off. The all-reduce push (`ar_mode` 2) needs a multicast workspace and is not driven here: the cells run
    `ar_mode` 0, where `y` is the o_proj output and nothing else is written.

    The QKV rows are handed from the GEMV CTAs to the attention CTAs by polling for the 0xFFFF marker the
    previous call left: a GEMV output whose bf16 bits are 0xFFFF (a NaN) would never be seen as written. Finite
    inputs cannot produce it.
    """

    def __call__(
        self,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_weight_scale: torch.Tensor,
        qkv_out: torch.Tensor,
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
        torch.ops.trtllm.minimax_m3_attn_mega(
            xq,
            xsf,
            qkv_weight,
            qkv_weight_scale,
            qkv_out,
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
        xq: torch.Tensor,
        xsf: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_weight_scale: torch.Tensor,
        qkv_out: torch.Tensor,
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
    ) -> MegaState:
        """The composition: ``packed = bf16(dequant(xq) @ dequant(W_qkv)^T)``, the fused attention's reference on
        those rows, then ``y = bf16(mxfp8(output) @ dequant(W_o)^T)`` (fp64 accumulation, one rounding per GEMV).

        The workspace, the output buffers, the geometry, ``use_pdl`` and the all-reduce arguments are ignored.
        """
        t = xq.shape[0]
        xq_b, w_b = xq.view(torch.uint8), qkv_weight.view(torch.uint8)
        packed = mx_gemm(
            xq_b,
            unswizzle_scales(xsf, t, HIDDEN // MXFP8_BLOCK),
            w_b,
            unswizzle_scales(qkv_weight_scale, PACKED_WIDTH, HIDDEN // MXFP8_BLOCK),
        )
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
        return MegaState(packed, attn.output, attn.kv_data, attn.kv_scale, attn.idx_k, out_y)

    def is_valid(
        self,
        xq: torch.Tensor,
        xsf: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_weight_scale: torch.Tensor,
        qkv_out: torch.Tensor,
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
            xq.shape[0],
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
        assert xsf.numel() >= 128 * (HIDDEN // MXFP8_BLOCK), (
            "xsf must hold one 128-row tile of swizzled scales"
        )
        parity = workspace_parity(flags, GRID)
        assert bool((qkv_out.view(torch.int16)[parity] == -1).all()), (
            "qkv_out's buffer for this call is not at rest (0xFFFF): the attention would read stale QKV rows"
        )
        check_handoff_at_rest(mxfp8_data, mxfp8_scale, parity)
        assert ar_mode != 2 or (ar_workspace != 0 and ar_flags != 0), (
            "ar_mode 2 needs the exchange's addresses"
        )


m3_attn_mega = M3AttnMega()
