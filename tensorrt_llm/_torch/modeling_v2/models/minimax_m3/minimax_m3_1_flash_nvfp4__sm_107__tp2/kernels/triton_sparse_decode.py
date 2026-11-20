# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-M3.1 NVFP4 / sm_107 / tp2 target's block-sparse GQA decode: the family Triton sparse decode
(``kernels/triton_sparse_decode`` of the family package) with this target's settings (``TARGET``).

Long-context TP2 steps of ``lc_min_rows(decode_query_len)``..``LC_MAX_ROWS`` query rows go to the one-launch
``m3_attn_lc`` kernel (LC2, TP2WIDE_LC2); every other step takes the Triton producer + merge, whose merge emits the
MXFP8 o_proj hand-off (``m3_prequant.SLOT``) up to ``MergeMxfp8.TRITON_MAX_ROWS`` rows.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm.logger import logger

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
from ..m31_switches import m31_disabled, m31_enabled
from . import m3_attn_lc, m3_prequant, m3_report


class MergeMxfp8:
    """MXFP8 copy of the split-K merge output for the o_proj GEMM.

    From ``MERGE_MXFP8_MIN_TOKENS`` to ``TRITON_MAX_ROWS`` rows the Triton merge kernel (and up to ``LC_MAX_ROWS`` rows
    the LC2 kernel) writes the E4M3 rows and swizzled UE8M0 scales next to the BF16 output and publishes them to
    ``m3_prequant.SLOT``, so the MXFP8 o_proj skips its own quantization. One ``MAX_ROWS``-row buffer per width serves
    every layer: each o_proj consumes it before the next layer's decode rewrites it, in stream order.
    """

    # TP2WIDE_LC2_448: the buffers hold the widest LC2 step (448 rows: K7 at 56 requests, K3 at 112);
    # the Triton merge's own hand-off keeps its 128-row bound (TRITON_MAX_ROWS), so only LC2 steps use the extra rows.
    MAX_ROWS = 448  # capacity: fixed: the hand-off buffers' rows (the widest LC2 decode step)
    TRITON_MAX_ROWS = (
        128  # capacity: fixed: the Triton merge / fullctx hand-off's rows (decode steps)
    )

    def __init__(self):
        self._buffers = {}

    @classmethod
    def scale_bytes(cls, rows: int, width: int) -> int:
        """The swizzled UE8M0 scales of ``rows`` rows: whole 128-row tiles (a partial last tile is laid out in full)."""
        return (rows + 127) // 128 * 128 * (width // 32)

    def buffers(self, width: int, device: torch.device):
        key = (width, device)
        if key not in self._buffers:
            self._buffers[key] = (
                torch.empty((self.MAX_ROWS, width), dtype=torch.float8_e4m3fn, device=device),
                # Whole scale tiles: MAX_ROWS (448) is not a multiple of 128; the kernels write the last tile in
                # full.
                torch.zeros(
                    (self.scale_bytes(self.MAX_ROWS, width),), dtype=torch.uint8, device=device
                ),
            )
        return self._buffers[key]

    @staticmethod
    def handoff(mxfp8, rows: int):
        """The hand-off of a ``rows``-row step: its E4M3 rows and the 128-row scale tiles that cover them. The o_proj
        GEMM checks the swizzled scales against the step's padded rows, so the step gets the tiles that cover its rows,
        never the whole buffer."""
        q8, sf = mxfp8
        width = q8.shape[-1]
        return q8[:rows], sf[: MergeMxfp8.scale_bytes(rows, width)]


MERGE_MXFP8 = MergeMxfp8()


# The long-context decode (``_longattn_lc_decode``) in its engagement line (modeling's SPARSE_DECODE_LC path; the
# rows it takes are LC_MIN_ROWS..LC_MAX_ROWS, see TP2WIDE_LC2).
LC_PATH = "m3_sparse_decode_lc"
# LC_TP2 (opt-in, ``TRTLLM_M31_ENABLE_LC_TP2=1``): the Triton route does not log its engagement line for the rows of
# an LC-shaped step that the long-context decode (TP2WIDE_LC2 below) does not take. The dispatch itself is
# TP2WIDE_LC2's either way; LC_TP2 changes the log only. Read at import.
LC_TP2 = m31_enabled("LC_TP2")
# The engagement path of the LC_TP2-off route: an LC-shaped step (packed NVFP4 pages, FP8 q, one top-k chunk) that runs
# the Triton decode (``m3_report.taken``, one line per rank and row count).
LC_TRITON_PATH = "m3_sparse_decode_lc_tp2_triton"
# TP2WIDE_LC2 (TP2 WIDE rule, ported from the M3 sm_107 TP2 target): pure generation steps with a selected top-16 table
# (not the identity selection) at LC_MIN_ROWS..LC_MAX_ROWS query rows run LC2
# (``family_ops().minimax_m3_sm107_tp2_sparse_decode_lc`` through ``m3_attn_lc``): attention, split-K merge and
# the MXFP8 o_proj hand-off in one launch, with the layer's N1 self-KV prefetch (modeling's
# SPARSE_DECODE_LC_PREFETCH_SELF_KV). Steps of at most 32 rows keep this target's Triton producer + merge exactly as
# before (same code path, bit for bit). The M3 target's LC2 settings: log-domain recurrence (``linear_softmax=False``),
# no pre-wait block-table staging (``LC_PREWAIT``), PDL from the environment, MXFP8 hand-off on.
# TRTLLM_M31_DISABLE_TP2WIDE_LC2=1 restores the Triton route at every width (LC_KEYS = frozenset()). Read at import.
TP2WIDE_LC2 = not m31_disabled("TP2WIDE_LC2")
# The fewest query rows LC2 takes: past the 32-row band, where the Triton route stays.
LC_MIN_ROWS = 33  # capacity: fixed: TP2 WIDE rule floor (rows <= 32 keep the Triton route)
# The long-context decode's shapes (modeling's SPARSE_DECODE_LC keys: "mxfp8" = a step of LC_MIN_ROWS..
# LC_MAX_ROWS rows, the hand-off written and published to m3_prequant.SLOT; "plain", a wider step, is not covered).
# frozenset() routes every step back to the Triton producer + merge.
LC_KEYS = frozenset({"mxfp8"}) if TP2WIDE_LC2 else frozenset()
# LC2_ALLROWS (not bitwise): LC2 also takes the steps outside the LC band, from one row
# (1.03x the Triton route at 8 rows) and past LC_MAX_ROWS (1.31x at 512 rows), in the same error band vs float64.
# Steps past LC_MAX_ROWS are the "plain" key: no MXFP8 hand-off (the o_proj quantizes its input as usual).
# TRTLLM_M31_DISABLE_LC2_ALLROWS=1 restores the lc_min_rows..LC_MAX_ROWS band. Read at import.
LC2_ALLROWS = TP2WIDE_LC2 and not m31_disabled("LC2_ALLROWS")
if LC2_ALLROWS:
    LC_KEYS = LC_KEYS | {"plain"}
# LC2_IDENTITY (not bitwise): an identity-selection step (every request within 16 blocks) also runs LC2,
# on the step's explicit identity table (2.0-2.4x the Triton route, the same band). TRTLLM_M31_DISABLE_LC2_IDENTITY=1
# keeps those steps on the Triton route. Read at import.
LC2_IDENTITY = TP2WIDE_LC2 and not m31_disabled("LC2_IDENTITY")
# TP2WIDE_LC2_448 (exact-class as the 33..128 band): LC2 takes up to 448 query rows (agg TP2 c46 / c56 /
# c64 at K3 = 184 / 224 / 256 rows; K7 c46 / c56 = 368 / 448) in one launch with the MXFP8 hand-off, instead of the
# Triton producer + merge (no hand-off past 128 rows) there. TRTLLM_M31_DISABLE_TP2WIDE_LC2_448=1 restores the 128-row
# bound (wider steps back on the Triton route). Read at import.
LC_MAX_ROWS = (  # capacity: fixed: LC2's widest step
    MergeMxfp8.TRITON_MAX_ROWS if m31_disabled("TP2WIDE_LC2_448") else MergeMxfp8.MAX_ROWS
)

# A step of 8 query rows per request (draft length 7) uses LC2 from 9 rows (not bitwise). Draft length 3 (4 query rows
# per request) does too with the opt-in TRTLLM_M31_ENABLE_TP2_LC2_SMALL_DRAFT_LEN_3=1.
# TRTLLM_M31_DISABLE_TP2_LC2_SMALL=1 restores the 33-row lower bound for both. Other query lengths keep the 33-row lower
# bound. Read once at import.
TP2_LC2_SMALL = not m31_disabled("TP2_LC2_SMALL")
LC2_SMALL_QLENS = frozenset({4, 8} if m31_enabled("TP2_LC2_SMALL_DRAFT_LEN_3") else {8})
LC2_SMALL_MIN_ROWS = 9  # capacity: fixed: measured crossover (12 rows, the first K3 step past 8)


def lc_min_rows(decode_query_len: int) -> int:
    """The fewest query rows LC2 takes for a step of ``decode_query_len`` query rows per request."""
    if LC2_ALLROWS:
        return 1
    if TP2_LC2_SMALL and int(decode_query_len) in LC2_SMALL_QLENS:
        return LC2_SMALL_MIN_ROWS
    return LC_MIN_ROWS


# LC2 stages the request's block-table row before its PDL grid wait: off on the M3 target (measured +1.6..+1.8 us at 32
# rows; flat to -0.2 us below), up to LC_PREWAIT_MAX_ROWS query rows when on (never at LC_MIN_ROWS and above).
LC_PREWAIT = False
LC_PREWAIT_MAX_ROWS = 16  # capacity: fixed: measured occupancy crossover
# The once-per-row-count engagement line of TP2WIDE_LC2 (per process; every GEN rank logs its own).
_TP2WIDE_LC2_LOGGED: set = set()


def _log_tp2wide_lc2(rows: int) -> None:
    if rows not in _TP2WIDE_LC2_LOGGED:
        _TP2WIDE_LC2_LOGGED.add(rows)
        logger.info(f"[M3.1 tp2] TP2WIDE_LC2 taken: {rows} rows")


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
) -> None:
    """minimax_m3_sparse_attn_decode through m3_attn_lc (LC2): same table, output and MXFP8 hand-off contract."""
    use_pdl = get_env_enable_pdl()
    total_q, num_heads, head_dim = q.shape
    prewait = bool(LC_PREWAIT) and total_q <= LC_PREWAIT_MAX_ROWS
    mxfp8 = None
    m3_prequant.SLOT.clear()
    if MERGE_MXFP8_MIN_TOKENS <= total_q <= LC_MAX_ROWS:
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
        linear_softmax=False,
        mxfp8=mxfp8,
        use_pdl=use_pdl,
        prewait=prewait,
        n1_ids=m3_attn_lc.N1_IDS.take(),
    )
    if prewait:
        m3_report.taken(LC_PREWAIT_PATH, total_q)
    if mxfp8 is not None:
        m3_prequant.SLOT.publish(
            output.view(total_q, num_heads * head_dim), *MergeMxfp8.handoff(mxfp8, total_q)
        )


# Engagement path of LC2's pre-wait block-table staging (LC_PREWAIT, up to its row cap).
LC_PREWAIT_PATH = "sparse_decode_lc_prewait"


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
    # Long-context TP2 steps (TP2WIDE_LC2): attention, split-K merge and MXFP8 hand-off in one launch. A selected table
    # on an NVFP4 cache with FP8 q at the TP2 heads and lc_min_rows(decode_query_len)..LC_MAX_ROWS rows is the
    # kernel's (modeling's SPARSE_DECODE_LC); operands it declines are a miss. Every other step (narrower, identity
    # selection, wider than LC_MAX_ROWS, or the kill switch) takes the unchanged Triton route below.
    lc_shaped = kv_nvfp4 and widen_q and num_topk_chunks is None
    lc_key = "mxfp8" if MERGE_MXFP8_MIN_TOKENS <= total_q <= LC_MAX_ROWS else "plain"
    lc_covered = (
        lc_shaped
        and (not identity_blocks or LC2_IDENTITY)
        and total_q >= lc_min_rows(decode_query_len)
        and num_heads == 32
        and num_kv_heads == 2
        and lc_key in LC_KEYS
    )
    if lc_covered and not m3_attn_lc.lc_applies(
        q, k_paged, v_paged, topk_idx, k_block_scale, v_block_scale, output
    ):
        m3_report.missed(
            LC_PATH,
            total_q,
            "the step's q / KV pages / scales / top-k table / output are not the layout the kernel is built for",
        )
        lc_covered = False
    if lc_covered:
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
        )
        m3_report.taken(LC_PATH, total_q)
        _log_tp2wide_lc2(total_q)
        return True
    if not LC_TP2 and lc_shaped:
        m3_report.taken(LC_TRITON_PATH, total_q, "LC rows on the Triton sparse decode (LC_TP2 off)")
    return False


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
        if MERGE_MXFP8_MIN_TOKENS <= total_q <= MergeMxfp8.TRITON_MAX_ROWS
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
    handoff=MergeMxfp8.handoff,
    cache_capability=True,
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
