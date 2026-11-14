# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_LC2: the sparse layers' generation decode attention as one LC2 launch at the attention-DP geometry.

Attention DP keeps all 4 KV heads of the model on every rank. ``family_ops().minimax_m3_sm107_tp2_sparse_decode_lc``
(cpp/tensorrt_llm/thop/minimaxM3/minimaxM3Sm107Tp2SparseDecodeLc2Op.cpp; its kernels take the KV-head count from the
cache view) replaces the in-tree Triton producer + merge (``minimax_m3_sparse_attn_decode``) through each layer's
``MsaDecodeFmha.sparse_decode_hook``: the split grid at 33..448 rows and the 16-CTA cluster below that, the log-domain
recurrence, the MXFP8 o_proj hand-off when ``merge_mxfp8`` reserved its buffers (DEP_MERGE_MXFP8), no pre-wait staging
and no N1 prefetch.
The split grid (33+ rows) launches with PDL; the cluster grid launches plainly. ``TRTLLM_M31_DISABLE_DEP_LC2_PDL=1``
launches the split plainly too; neither launch changes a value. Not bitwise the Triton route: both stay within the
float64-reference band at 8..448 rows of 8-row verify steps over 32k..220k-token and mixed contexts, and the split is
bitwise the cluster.

``dep_lc2_decode`` takes a call with E4M3 q, the packed-NVFP4 cache at 4 KV heads, BF16 output, decode_query_len 8
and at most 448 rows (``decline_reason``); any other call raises with the reason, as there is no Triton fallback. The
generation steps are always taken: max_num_tokens 384 caps a step at 384 rows (48 requests), and the DSpark verify with
7 draft tokens reads 8 rows per request. ``TRTLLM_M31_DISABLE_DEP_LC2=1`` installs no hook (the stock decode on every
call, the reference arm). With the switch on, a build without the op fails at ``install``. Read at import.

A taken call reads its KV lengths through the model's pad-row source when one is set (``set_pad_rows``, from
``pad_mask.arm_lc2``): CUDA-graph padding requests then get length 0, so the kernel selects no block for their rows and
writes zeros there (an empty row's merge weights are all 0). The stock decode keeps the real lengths.

The split's workspace (448 rows x 4 KV heads, ~237 MB per device) is reserved by ``reserve_workspace`` once the weights
are real and before the executor sizes its KV pool; a build whose reserve refuses 4 KV heads fails there, naming the
switch.

DEP_OWNT: a call LC2 takes runs OWN-T, the block-owner decode, instead
(``native_ops.minimax_m3_sm107_sparse_decode_own_t``,
cpp/tensorrt_llm/thop/minimaxM3/minimaxM3Sm107SparseDecodeOwnTOp.cpp): each block a request's rows selected is
dequantized once for all of them, and the step's blocks are split into equal weighted shares over one CTA per SM, in
three launches (union, main, merge) that take PDL as the split does (``TRTLLM_M31_DISABLE_DEP_LC2_PDL=1`` launches them
plainly). Not bitwise LC2: both stay within the float64-reference band at 8-row verify steps, and OWN-T is deterministic
for a given step but not batch-invariant (a request's rows may be split differently in another batch). Given the
hand-off buffers it writes the output's MXFP8 form as LC2's merge does (bitwise ``trtllm::mxfp8_quantize`` of its BF16
rows), and the call publishes it in ``merge_mxfp8.SLOT`` the same way. A call it declines (``own_t_decline_reason``)
runs LC2; ``TRTLLM_M31_DISABLE_DEP_OWNT=1`` runs LC2 on every call LC2 takes. Its workspace (256 items at the device's
SM count, ~30 MB) is reserved by ``reserve_own_t_workspace`` beside LC2's."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (
    native_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm.logger import logger

from . import merge_mxfp8 as _mx
from .m31_switches import m31_disabled

DEP_LC2 = not m31_disabled("DEP_LC2")
#: PDL on the split grid's attention and merge launches. The split reads seq_lens and q before its griddepcontrol.wait
#: (sparseDecodeLc2Split.cu:197-223); that holds while every kernel between the producer and LC2 (index score, select)
#: releases its dependents only at exit, as both do. Pre-wait staging stays off (its larger shared-memory footprint
#: costs the split more than it hides), and the 8-row cluster launches without PDL.
DEP_LC2_PDL = not m31_disabled("DEP_LC2_PDL")
DEP_OWNT = not m31_disabled("DEP_OWNT")
# DEP_OWNT_FASTPLAN: OWN-T's main kernel derives the union's fixed block weights in its plan instead of loading them
# (the same split, bitwise). Kill switch TRTLLM_M31_DISABLE_DEP_OWNT_FASTPLAN=1, passed as the op's fast_plan argument.
OWNT_FASTPLAN = not m31_disabled("DEP_OWNT_FASTPLAN")
# TP2WIDE_LC2_SPLIT: steps of 33..448 query rows run LC2 without its thread-block cluster (the per-(row, block) work
# split over CTAs, then merged), bit for bit the cluster kernel. Kill switch TRTLLM_M31_DISABLE_TP2WIDE_LC2_SPLIT=1
# runs the cluster kernel at every row count and reserves no split workspace.
LC2_SPLIT = not m31_disabled("TP2WIDE_LC2_SPLIT")
#: The first step the split grid takes; narrower steps run the cluster grid.
SPLIT_MIN_ROWS = 33

#: The geometry the route takes: 16 query heads on each of the rank's 4 KV heads, head_dim 128, page 128, top-k 16.
KV_HEADS = 4
HEADS_PER_KV = 16
HEAD_DIM = 128
PAGE = 128
TOPK = 16
#: The verify step with 7 draft tokens (7 + 1 rows per request), the only query length this geometry takes.
DECODE_QUERY_LEN = 8
#: The widest step the split's workspace is reserved for; a wider step raises (generation steps are at most 384 rows).
MAX_ROWS = 448  # capacity: fixed: LC2's widest supported step

#: OWN-T's widest step: (request, KV head) items.
OWNT_MAX_ITEMS = 256  # capacity: fixed: the kernel's plan holds 256 items

_OP_NAME = "minimax_m3_sm107_tp2_sparse_decode_lc"
_PREPARE_NAME = "minimax_m3_sm107_tp2_sparse_decode_lc_prepare"
_OWNT_NAME = "minimax_m3_sm107_sparse_decode_own_t"
_OWNT_PREPARE_NAME = "minimax_m3_sm107_sparse_decode_own_t_prepare"
# The (outcome, detail) pairs already logged: one line per distinct reason a call is taken or declined.
_LOGGED: set = set()
# The model's pad-row source (``pad_mask.PadMask``), or None: a taken call reads its KV lengths through it.
_PAD_ROWS = None


def set_pad_rows(source) -> None:
    """Route every taken call's KV lengths through ``source.lc2_seq_lens``; None reads them as given."""
    global _PAD_ROWS
    _PAD_ROWS = source


def lc2_op():
    """``family_ops().minimax_m3_sm107_tp2_sparse_decode_lc`` when this build carries it, else None."""
    return getattr(family_ops(), _OP_NAME, None)


def own_t_op():
    """``native_ops.minimax_m3_sm107_sparse_decode_own_t`` when this build carries it, else None."""
    return getattr(native_ops, _OWNT_NAME, None)


def _misaligned(t: torch.Tensor, align: int) -> bool:
    return t.data_ptr() % align != 0 or t.stride(0) % align != 0 or t.stride(1) % align != 0


def decline_reason(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    decode_query_len: int,
    kv_scale: Optional[torch.Tensor],
    k_block_scale: Optional[torch.Tensor],
    v_block_scale: Optional[torch.Tensor],
    k_global_scale: Optional[torch.Tensor],
    v_global_scale: Optional[torch.Tensor],
    num_topk_chunks: Optional[int],
) -> Optional[str]:
    """Why LC2 does not take this decode call, or None when it does. Reads shapes, dtypes and strides only (no
    device read), so it is safe under graph capture. The kernel ignores the row pitch of the cache, its scales and q,
    and reads the block table and seq_lens unchecked, so those layouts are checked here."""
    nvfp4 = (k_block_scale, v_block_scale, k_global_scale, v_global_scale)
    if kv_scale is not None or num_topk_chunks is not None or any(t is None for t in nvfp4):
        return "not the packed-NVFP4 cache call"
    if q.dtype != torch.float8_e4m3fn:
        return f"q is {q.dtype}, not E4M3"
    if decode_query_len != DECODE_QUERY_LEN:
        return f"decode_query_len {decode_query_len}"
    total_q = q.shape[0] if q.dim() == 3 else -1
    if not 1 <= total_q <= MAX_ROWS:
        return f"{total_q} rows (outside 1..{MAX_ROWS})"
    if seq_lens.dim() != 1 or seq_lens.shape[0] * decode_query_len != total_q:
        return "q does not cover exactly the batch's decode span"
    heads = KV_HEADS * HEADS_PER_KV
    if (
        tuple(q.shape[1:]) != (heads, HEAD_DIM)
        or q.stride(2) != 1
        or _misaligned(q, 16)
        or output.dtype != torch.bfloat16
        or tuple(output.shape) != (total_q, heads, HEAD_DIM)
        or output.stride(2) != 1
        or output.stride(1) != HEAD_DIM
    ):
        return "q / output not [T, 64, 128] with dense rows (E4M3 q, BF16 output)"
    for t in (k_paged, v_paged):
        if (
            t.dim() != 4
            or t.element_size() != 1
            or tuple(t.shape[1:]) != (KV_HEADS, PAGE, HEAD_DIM // 2)
            or t.stride(3) != 1
            or t.stride(2) != HEAD_DIM // 2
            or _misaligned(t, 16)
        ):
            return "cache not [pages, 4, 128, 64] packed bytes with dense rows"
    if v_paged.shape != k_paged.shape or v_paged.stride() != k_paged.stride():
        return "k and v views differ"
    for t in (k_block_scale, v_block_scale):
        if (
            t.dim() != 4
            or t.element_size() != 1
            or tuple(t.shape[1:]) != (KV_HEADS, PAGE, HEAD_DIM // 16)
            or t.stride(3) != 1
            or t.stride(2) != HEAD_DIM // 16
            or _misaligned(t, 8)
        ):
            return "block scales not [pages, 4, 128, 8] bytes with dense rows"
    if v_block_scale.stride() != k_block_scale.stride():
        return "k and v block scales differ"
    for t in (k_global_scale, v_global_scale):
        if t.dtype != torch.float32 or t.numel() != 1:
            return "global scales not one float32 each"
    if (
        topk_idx.dtype != torch.int32
        or tuple(topk_idx.shape) != (KV_HEADS, total_q, TOPK)
        or block_table.dtype != torch.int32
        or block_table.dim() != 2
        or block_table.stride(1) != 1
        or seq_lens.dtype != torch.int32
        or seq_lens.stride(0) != 1
    ):
        return "top-k table / block table / seq_lens layout"
    return None


def own_t_decline_reason(
    q: torch.Tensor,
    k_block_scale: torch.Tensor,
    v_block_scale: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
) -> Optional[str]:
    """Why OWN-T does not take a call LC2 takes, or None when it does: beyond LC2's layouts, OWN-T bulk-copies each
    page's block scales (16-byte aligned views) and the main kernel stores 8 output elements at a time (16-byte
    aligned rows). Reads shapes, dtypes and strides only."""
    if not q.is_cuda:
        return "not a CUDA call"
    if seq_lens.shape[0] * KV_HEADS > OWNT_MAX_ITEMS:
        return f"{seq_lens.shape[0]} requests (more than {OWNT_MAX_ITEMS // KV_HEADS})"
    if any(_misaligned(t, 16) for t in (k_block_scale, v_block_scale)):
        return "block scales not 16-byte aligned"
    if output.data_ptr() % 16 != 0 or output.stride(0) % 8 != 0:
        return "output rows not 16-byte aligned"
    return None


def _log_once(taken: bool, detail: str, item: str = "DEP_LC2", other: str = "declined") -> None:
    if (item, taken, detail) not in _LOGGED:
        _LOGGED.add((item, taken, detail))
        logger.info(f"[M3.1 DEP] {item} {'taken' if taken else other}: {detail}")


def dep_lc2_decode(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    kv_scale: Optional[torch.Tensor] = None,
    k_block_scale: Optional[torch.Tensor] = None,
    v_block_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[torch.Tensor] = None,
    v_global_scale: Optional[torch.Tensor] = None,
    num_topk_chunks: Optional[int] = None,
) -> None:
    """The layers' ``sparse_decode_hook``: ``minimax_m3_sparse_attn_decode``'s arguments and in-place output, on LC2
    when ``decline_reason`` finds none, else a RuntimeError naming the reason; a call LC2 takes runs OWN-T when
    DEP_OWNT is on, the op is built and ``own_t_decline_reason`` finds none. A call LC2 or OWN-T runs with the hand-off
    buffers reserved also publishes the output's MXFP8 form in ``merge_mxfp8.SLOT``; every call first clears it."""
    _mx.SLOT.clear()
    op = lc2_op()
    reason = (
        "the op is not built"
        if op is None
        else decline_reason(
            q,
            k_paged,
            v_paged,
            topk_idx,
            block_table,
            seq_lens,
            output,
            decode_query_len,
            kv_scale,
            k_block_scale,
            v_block_scale,
            k_global_scale,
            v_global_scale,
            num_topk_chunks,
        )
    )
    if reason is not None:
        raise RuntimeError(
            f"[M3.1 DEP] DEP_LC2 cannot take this sparse decode call: {reason}. The DEP target has no Triton "
            "fallback; TRTLLM_M31_DISABLE_DEP_LC2=1 runs the stock decode on every call"
        )
    total_q = q.shape[0]
    width = KV_HEADS * HEADS_PER_KV * HEAD_DIM
    handoff = _mx.buffers(total_q, output.device)
    if DEP_OWNT:
        own_t = own_t_op()
        own_t_reason = (
            "the op is not built"
            if own_t is None
            else own_t_decline_reason(q, k_block_scale, v_block_scale, seq_lens, output)
        )
        if own_t_reason is None:
            _log_once(
                True,
                f"{total_q} rows" + (", MXFP8 o_proj hand-off" if handoff is not None else ""),
                item="DEP_OWNT",
            )
            if _PAD_ROWS is not None:
                seq_lens = _PAD_ROWS.lc2_seq_lens(seq_lens)
            out_2d = output.view(total_q, width)
            own_t(
                q.view(torch.uint8),
                k_paged,
                v_paged,
                k_block_scale,
                v_block_scale,
                k_global_scale,
                v_global_scale,
                topk_idx,
                block_table,
                seq_lens,
                out_2d,
                None if handoff is None else handoff[0],
                None if handoff is None else handoff[1],
                width // 128,
                decode_query_len,
                sm_scale,
                DEP_LC2_PDL,
                OWNT_FASTPLAN,
            )
            if handoff is not None:
                _mx.SLOT.publish(out_2d, *handoff)
            return
        _log_once(False, own_t_reason, item="DEP_OWNT", other="declined (LC2 runs)")
    split = total_q >= SPLIT_MIN_ROWS
    _log_once(
        True,
        f"{'split' if split else 'cluster'} grid at {total_q} rows"
        + (", MXFP8 o_proj hand-off" if handoff is not None else ""),
    )
    if _PAD_ROWS is not None:
        seq_lens = _PAD_ROWS.lc2_seq_lens(seq_lens)
    out_2d = output.view(total_q, width)
    op(
        q.view(torch.uint8),
        k_paged,
        v_paged,
        k_block_scale,
        v_block_scale,
        k_global_scale,
        v_global_scale,
        topk_idx,
        block_table,
        seq_lens,
        out_2d,
        None if handoff is None else handoff[0],
        None if handoff is None else handoff[1],
        width // 128,
        decode_query_len,
        sm_scale,
        False,
        DEP_LC2_PDL and split,
        False,
        None,
        LC2_SPLIT,
    )
    if handoff is not None:
        _mx.SLOT.publish(out_2d, *handoff)


def serves(attn: torch.nn.Module) -> bool:
    """Whether ``dep_lc2_decode`` is the decode hook of one of the attention module's MSA decode libraries."""
    manager = getattr(getattr(attn, "attn", None), "_fmha_manager", None)
    return any(
        getattr(fmha, "sparse_decode_hook", None) is dep_lc2_decode
        for fmha in getattr(manager, "fmha_libs", ())
    )


def install(layers: Iterable[torch.nn.Module]) -> int:
    """Set ``dep_lc2_decode`` as the ``sparse_decode_hook`` of each layer's MSA decode library that has none and runs
    4 KV heads; returns how many carry it, including any an earlier call set (0 with TRTLLM_M31_DISABLE_DEP_LC2=1).
    Raises on a build without the op, as the DEP target has no Triton fallback."""
    if not DEP_LC2:
        return 0
    if lc2_op() is None:
        raise RuntimeError(
            f"the MiniMax-M3.1 DEP target's sparse decode needs torch.ops.trtllm.{_OP_NAME}, which this build lacks "
            "(no Triton fallback); TRTLLM_M31_DISABLE_DEP_LC2=1 runs the stock decode"
        )
    count = 0
    for layer in layers:
        backend = getattr(getattr(layer, "self_attn", None), "attn", None)
        if getattr(backend, "num_kv_heads", None) != KV_HEADS:
            continue
        manager = getattr(backend, "_fmha_manager", None)
        for fmha in getattr(manager, "fmha_libs", ()):
            if getattr(fmha, "sparse_decode_hook", False) is None:
                fmha.sparse_decode_hook = dep_lc2_decode
            if getattr(fmha, "sparse_decode_hook", None) is dep_lc2_decode:
                count += 1
    return count


def reserve_workspace(device: torch.device) -> int:
    """Reserve the split's workspace for MAX_ROWS rows at KV_HEADS on ``device``, outside any graph capture; returns
    its bytes (0 when the build has no reserve or the split is disabled)."""
    prepare = getattr(family_ops(), _PREPARE_NAME, None)
    if prepare is None or not LC2_SPLIT:
        return 0
    anchor = torch.empty(0, dtype=torch.uint8, device=device)
    try:
        return int(prepare(anchor, KV_HEADS))
    except RuntimeError as err:
        raise RuntimeError(
            f"the MiniMax-M3.1 DEP target's LC2 decode needs a build whose {_PREPARE_NAME} reserves {KV_HEADS} KV "
            f"heads ({err}); TRTLLM_M31_DISABLE_DEP_LC2=1 keeps the stock decode"
        ) from err


def reserve_own_t_workspace(device: torch.device) -> int:
    """Prepare ``device`` for OWN-T and reserve its workspace for OWNT_MAX_ITEMS items, outside any graph capture;
    returns its bytes (0 with TRTLLM_M31_DISABLE_DEP_OWNT=1, without the op, or off CUDA). A build without an OWN-T
    kernel for the device's architecture fails there, naming the switch."""
    prepare = getattr(native_ops, _OWNT_PREPARE_NAME, None)
    if not DEP_OWNT or prepare is None or own_t_op() is None or device.type != "cuda":
        return 0
    anchor = torch.empty(0, dtype=torch.uint8, device=device)
    try:
        return int(prepare(anchor, OWNT_MAX_ITEMS))
    except RuntimeError as err:
        raise RuntimeError(
            f"the MiniMax-M3.1 DEP target's OWN-T decode could not prepare {device} ({err}); "
            "TRTLLM_M31_DISABLE_DEP_OWNT=1 runs LC2"
        ) from err


def own_t_status() -> str:
    """OWN-T's engagement, for the model's summary line."""
    if not DEP_OWNT:
        return "off (TRTLLM_M31_DISABLE_DEP_OWNT=1)"
    return "on" if own_t_op() is not None else "not built"
