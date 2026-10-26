# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LC2_MIXED (not bitwise): a mixed ctx + gen step's sparse generation rows (the
MsaDecodeFmha sparse_decode_hook) run LC2 (LC2-split: the C++ one-launch sparse decode) instead of GQAMIX's Triton
copy: 1.3-1.9x faster at 16..224 rows, in the same error band vs float64 as the Triton route. No MXFP8 hand-off: the
mixed step's o_proj input also holds the context rows. Operands the kernel declines (``m3_attn_lc.lc_applies``) keep
the Triton copy. Kill switch: TRTLLM_M31_DISABLE_LC2_MIXED=1, read at import.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm.logger import logger

from ..m31_switches import m31_disabled
from . import m3_attn_lc, m3_report
from .triton_sparse_decode import TP2WIDE_LC2

LC2_MIXED = TP2WIDE_LC2 and not m31_disabled("LC2_MIXED")
_LOGGED: set = set()


def lc2_mixed_decode(
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
    kv_scale=None,
    k_block_scale=None,
    v_block_scale=None,
    k_global_scale=None,
    v_global_scale=None,
    num_topk_chunks=None,
    identity_blocks=False,
) -> bool:
    """Run ``minimax_m3_sparse_attn_decode``'s contract through LC2 when the operands fit; return whether it ran."""
    if not LC2_MIXED or kv_scale is not None or num_topk_chunks is not None:
        return False
    if any(t is None for t in (k_block_scale, v_block_scale, k_global_scale, v_global_scale)):
        return False
    if q.dtype != torch.float8_e4m3fn or int(q.shape[0]) == 0:
        return False
    if not m3_attn_lc.lc_applies(
        q, k_paged, v_paged, topk_idx, k_block_scale, v_block_scale, output
    ):
        return False  # operands outside LC2's layout: GQAMIX's Triton copy runs them
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
        mxfp8=None,
        use_pdl=get_env_enable_pdl(),
    )
    rows = int(q.shape[0])
    m3_report.work("LC2_MIXED", "mixed-step sparse generation rows on LC2")
    if rows not in _LOGGED:
        _LOGGED.add(rows)
        logger.info(f"[M3.1 tp2] LC2_MIXED taken: {rows} mixed-step generation rows")
    return True
