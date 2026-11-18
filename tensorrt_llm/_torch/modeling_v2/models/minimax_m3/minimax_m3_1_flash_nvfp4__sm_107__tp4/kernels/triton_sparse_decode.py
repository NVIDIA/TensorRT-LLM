# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3.1 NVFP4 / sm_107 / tp4 target's block-sparse GQA decode: the family Triton sparse decode
(``kernels/triton_sparse_decode`` of the family package) with this target's settings (``TARGET``).

Long-context TP4 steps go to the one-launch ``m3_attn_lc`` kernel; every other step takes the Triton producer +
merge, whose merge emits the MXFP8 o_proj hand-off (``m3_prequant.SLOT``) up to ``MergeMxfp8.MAX_ROWS`` rows.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ...kernels import triton_sparse_decode as _family
from ...kernels.triton_sparse_decode import (
    DEVICE_IDENTITY_PATH,
    MERGE_MXFP8_MIN_TOKENS,
    MQ_MIN_TOKENS,
    MQ_NUM_WARPS,
    NVFP4_SF_VEC_SIZE,
    SPARSE_BLOCK_SIZE,
    _sm100f_nvfp4_merge_launch_options,
    _sm100f_nvfp4_num_topk_chunks,
    _sm100f_nvfp4_query_group_size,
    _sm100f_nvfp4_use_linear_softmax,
    resolve_num_topk_chunks,
)
from . import m3_attn_lc, m3_prequant, m3_report


class MergeMxfp8:
    """MXFP8 copy of the split-K merge output for the o_proj GEMM.

    From ``MERGE_MXFP8_MIN_TOKENS`` to ``MAX_ROWS`` rows the merge kernel writes the E4M3 rows and swizzled UE8M0
    scales next to the BF16 output and publishes them to ``m3_prequant.SLOT``, so the MXFP8 o_proj skips its own
    quantization. One 128-row buffer per width serves every layer: each o_proj consumes it before the next layer's
    merge rewrites it, in stream order.
    """

    MAX_ROWS = 128  # capacity: fixed: the merge hand-off's rows (decode steps)

    def __init__(self):
        self._buffers = {}

    def buffers(self, width: int, device: torch.device):
        key = (width, device)
        if key not in self._buffers:
            self._buffers[key] = (
                torch.empty((self.MAX_ROWS, width), dtype=torch.float8_e4m3fn, device=device),
                torch.zeros((self.MAX_ROWS * (width // 32),), dtype=torch.uint8, device=device),
            )
        return self._buffers[key]


MERGE_MXFP8 = MergeMxfp8()


# The long-context decode (``_longattn_lc_decode``) in its engagement line (no row range: every step the LC
# applies to takes it).
LC_PATH = "m3_sparse_decode_lc"


def _longattn_lc_decode(
    q,
    k_paged,
    v_paged,
    topk_idx,
    block_table,
    seq_lens,
    *,
    sm_scale,
    output,
    decode_query_len,
    k_block_scale,
    v_block_scale,
    k_global_scale,
    v_global_scale,
    identity=False,
    identity_step=False,
) -> None:
    """minimax_m3_sparse_attn_decode through m3_attn_lc: same table, output and MXFP8 hand-off contract. ``identity``
    takes the short-sequence identity selection from the device lengths (the table is not read); ``identity_step``
    marks a step with no select (the kernel reads q after its grid wait), also when the staged table is read."""

    use_pdl = get_env_enable_pdl()
    total_q, num_heads, head_dim = q.shape
    linear = _sm100f_nvfp4_use_linear_softmax(
        total_q=total_q,
        num_kv_heads=1,
        gqa_group_size=num_heads,
        max_topk=int(topk_idx.shape[-1]),
        decode_query_len=decode_query_len,
    )
    mxfp8 = None
    m3_prequant.SLOT.clear()
    if MERGE_MXFP8_MIN_TOKENS <= total_q <= MergeMxfp8.MAX_ROWS:
        mxfp8 = MERGE_MXFP8.buffers(num_heads * head_dim, output.device)
    # The early trigger lets the o_proj launch and stream its weights while attention runs.
    m3_attn_lc.m3_lc_attn(
        q,
        k_paged,
        v_paged,
        topk_idx,
        block_table,
        seq_lens,
        sm_scale=sm_scale,
        output=output,
        decode_query_len=decode_query_len,
        k_block_scale=k_block_scale,
        v_block_scale=v_block_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        linear_softmax=linear,
        mxfp8=mxfp8,
        use_pdl=use_pdl,
        early_trigger=True,
        # This target's op (sparseDecodeLc4): the merge weights formed once per CTA, and up to
        # m3_attn_lc.ONE_DIV_MAX_ROWS rows the split-K merge dividing once per output (fixed order, no atomics;
        # inside the same band of the float64 reference).
        merge_once=True,
        prefetch=m3_attn_lc.PREFETCH.take(),
        next_kv=m3_attn_lc.NEXT_KV.take(),
        identity=identity,
        identity_step=identity_step,
        one_div=m3_attn_lc.one_div_applies(total_q),
    )
    if m3_attn_lc.one_div_applies(total_q):
        m3_report.taken(ONE_DIV_MERGE_PATH, total_q)
    if mxfp8 is not None:
        m3_prequant.SLOT.publish(
            output.view(total_q, num_heads * head_dim), mxfp8[0][:total_q], mxfp8[1]
        )


# Engagement path of the LC's one-division split-K merge.
ONE_DIV_MERGE_PATH = "sparse_decode_lc_one_div_merge"


def _long_context(
    q,
    k_paged,
    v_paged,
    topk_idx,
    block_table,
    seq_lens,
    *,
    sm_scale,
    output,
    decode_query_len,
    k_block_scale,
    v_block_scale,
    k_global_scale,
    v_global_scale,
    kv_nvfp4,
    widen_q,
    num_topk_chunks,
    identity_blocks,
    total_q,
    num_heads,
    num_kv_heads,
) -> bool:
    """This target's long-context route of ``minimax_m3_sparse_attn_decode`` (the family module's
    ``SparseDecodeTarget.long_context``): runs the step and returns True when the route takes it."""
    # Long-context TP4 steps: attention, split-K merge and MXFP8 hand-off in one launch.

    if (
        kv_nvfp4
        and widen_q
        and num_topk_chunks is None
        and m3_attn_lc.lc_applies(
            q, k_paged, v_paged, topk_idx, k_block_scale, v_block_scale, output
        )
    ):
        _longattn_lc_decode(
            q,
            k_paged,
            v_paged,
            topk_idx,
            block_table,
            seq_lens,
            sm_scale=sm_scale,
            output=output,
            decode_query_len=decode_query_len,
            k_block_scale=k_block_scale,
            v_block_scale=v_block_scale,
            k_global_scale=k_global_scale,
            v_global_scale=v_global_scale,
            # The identity selection from the device lengths (the staged table is not read).
            identity=identity_blocks,
            # Every identity step (no select ran): q after the grid wait.
            identity_step=identity_blocks,
        )
        m3_report.taken(LC_PATH, total_q)
        return True
    return False


def _handoff(mxfp8, rows: int):
    """The published hand-off of a ``rows``-row step: its E4M3 rows and the whole scale buffer."""
    return mxfp8[0][:rows], mxfp8[1]


def _identity_producer(total_q: int, use_mq: bool, num_kv_heads: int) -> None:
    """The engagement of an identity-selection step's producer variant."""
    # The identity-selection route is modeling.SPARSE_DECODE_PRODUCER's.
    m3_report.taken(
        "sparse_decode_producer",
        total_q,
        "multi_query" if use_mq else "per_row",
    )


def _triton_mxfp8(total_q: int, head_dim: int, width: int, output: torch.Tensor):
    """The hand-off buffers the Triton merge writes for a step, or None."""
    return (
        MERGE_MXFP8.buffers(width, output.device)
        if MERGE_MXFP8_MIN_TOKENS <= total_q <= MergeMxfp8.MAX_ROWS
        and head_dim == 128
        and output.dim() == 3
        and output.is_contiguous()
        and output.dtype == torch.bfloat16
        else None
    )


#: This target's settings of the family sparse decode.
TARGET = _family.SparseDecodeTarget(
    report=m3_report,
    prequant=m3_prequant,
    long_context=_long_context,
    identity_producer=_identity_producer,
    triton_mxfp8=_triton_mxfp8,
    handoff=_handoff,
    cache_capability=False,
)

#: ``kernels/triton_sparse_decode.minimax_m3_sparse_attn_decode`` of the family package with this target's ``TARGET``.
minimax_m3_sparse_attn_decode = functools.partial(
    _family.minimax_m3_sparse_attn_decode, target=TARGET
)


__all__ = [
    "DEVICE_IDENTITY_PATH",
    "MERGE_MXFP8",
    "MERGE_MXFP8_MIN_TOKENS",
    "MQ_MIN_TOKENS",
    "MQ_NUM_WARPS",
    "MergeMxfp8",
    "NVFP4_SF_VEC_SIZE",
    "SPARSE_BLOCK_SIZE",
    "TARGET",
    "_sm100f_nvfp4_merge_launch_options",
    "_sm100f_nvfp4_num_topk_chunks",
    "_sm100f_nvfp4_query_group_size",
    "_sm100f_nvfp4_use_linear_softmax",
    "minimax_m3_sparse_attn_decode",
    "resolve_num_topk_chunks",
]
