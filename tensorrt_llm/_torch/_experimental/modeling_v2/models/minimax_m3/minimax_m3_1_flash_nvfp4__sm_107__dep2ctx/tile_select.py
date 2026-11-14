# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_TSEL (#20629 TILE_SELECT) on the sm_107 dep2ctx target: the prefill top-16 block select as the tile
select.

``minimax_m31_select_prefill_tile`` reads each 32-byte sector of the head-major proxy scores once for 8 queries of a KV
head; the built-in ``minimax_m3_select_blocks`` reads one score per sector per pass. Its table equals the built-in one
bit for bit on sm_107. Registered on every layer's MSA indexer
``selector_hook``; steps outside its rows / blocks range, graph captures and builds without the op keep the built-in
select.

Kill switch: ``TRTLLM_M31_DISABLE_DEP2CTX_TSEL=1``.

DEP2CTX_TSEL_ROWS (exact): the exact Q8KV8 prefill scorer's scores (its [T * H, pitch] buffer as a zero-copy
[H, pages, T] view with unit block stride, idx_scorer) take ``minimax_m31_select_prefill_tile_rows``, the same kernel
with a row-major staging: the shared memory its unchanged select phase reads is the same, so the table is the tile op's
on the same values bit for bit. ``select_rows`` is registered as
the indexer's ``prefill_scorer_selector``, which only the scorer's steps reach: ``select`` (every other select, the
NVFP4 decode reader's row-major scores on mixed / eager decode rows included) still takes head-major scores only, so
those rows keep their DEP_VLLM_TOPK / built-in select. Kill switch ``TRTLLM_M31_DISABLE_DEP2CTX_TSEL_ROWS=1``
(idx_scorer does not register without the rows op, so the scorer is off too).
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
)
from tensorrt_llm.logger import logger

from .m31_switches import m31_disabled

TSEL = not m31_disabled("DEP2CTX_TSEL")
TSEL_ROWS = TSEL and not m31_disabled("DEP2CTX_TSEL_ROWS")
TSEL_MIN_ROWS = 256  # #20629's bound: below it the built-in one-CTA-per-row grid serves the select
TSEL_MAX_BLOCKS = 5856  # the op's shared-memory bound (kMinimaxM3SelectPrefillTileMaxBlocks)

_STATE: dict = {}


def _op():
    """``family_ops().minimax_m31_select_prefill_tile`` on sm_107, or None. Decided once per process."""
    if "op" not in _STATE:
        op = None
        if torch.cuda.get_device_capability() == (10, 7):
            op = getattr(family_ops(), "minimax_m31_select_prefill_tile", None)
        _STATE["op"] = op
        logger.info(
            "[M3.1 dep2ctx] DEP2CTX_TSEL prefill tile select: "
            f"{'engaged' if op is not None else 'not taken: not sm_107 or the op is not built'} "
            f"(rows >= {TSEL_MIN_ROWS}, blocks <= {TSEL_MAX_BLOCKS})"
        )
    return _STATE["op"]


def _rows_op():
    """``family_ops().minimax_m31_select_prefill_tile_rows`` on sm_107 (DEP2CTX_TSEL_ROWS), or None. Decided once
    per process."""
    if "rows_op" not in _STATE:
        op = None
        if TSEL_ROWS and torch.cuda.get_device_capability() == (10, 7):
            op = getattr(family_ops(), "minimax_m31_select_prefill_tile_rows", None)
        _STATE["rows_op"] = op
        logger.info(
            "[M3.1 dep2ctx] DEP2CTX_TSEL_ROWS row-major prefill tile select: "
            + (
                "engaged"
                if op is not None
                else "not taken: TRTLLM_M31_DISABLE_DEP2CTX_TSEL(_ROWS)=1"
                if not TSEL_ROWS
                else "not taken: not sm_107 or the op is not built"
            )
        )
    return _STATE["rows_op"]


def _in_range(max_score_kv) -> bool:
    """Whether the tile select's host-side gate admits these scores (any layout)."""
    return not (
        torch.cuda.is_current_stream_capturing()  # the host-side rows / blocks gate is not graph-stable
        or max_score_kv.dtype != torch.float32
        or max_score_kv.dim() != 3
        or max_score_kv.shape[2] < TSEL_MIN_ROWS
        or max_score_kv.shape[1] > TSEL_MAX_BLOCKS
    )


def select(max_score_kv, n_valid_blocks, init_blocks, local_blocks):
    """The MSA indexer's selector hook: the head-major [total_q, num_kv_heads, 16] table of
    ``select_blocks_from_maxscore``, or None (outside the tile select's rows / blocks range, not head-major, or no op)
    to keep it."""
    if not _in_range(max_score_kv) or max_score_kv.stride(2) != 1:
        return None
    return _run(_op(), max_score_kv, n_valid_blocks, init_blocks, local_blocks)


def select_rows(max_score_kv, n_valid_blocks, init_blocks, local_blocks):
    """The MSA indexer's prefill_scorer_selector (DEP2CTX_TSEL_ROWS): ``select``'s table for the prefill scorer's
    row-major scores (unit block stride), or None (outside the range, another layout, or no rows op) to leave them to
    the indexer's other selects."""
    if not _in_range(max_score_kv) or max_score_kv.stride(1) != 1:
        return None
    return _run(_rows_op(), max_score_kv, n_valid_blocks, init_blocks, local_blocks)


def _run(op, max_score_kv, n_valid_blocks, init_blocks, local_blocks):
    """The table ``op`` (the tile op or its row-major twin) selects, or None without the op."""
    if op is None:
        return None
    num_kv_heads, num_blocks, total_q = max_score_kv.shape
    out = torch.empty(
        (num_kv_heads, total_q, MSA_REQUIRED_TOPK), dtype=torch.int32, device=max_score_kv.device
    ).permute(1, 0, 2)
    nvb = n_valid_blocks.to(device=max_score_kv.device, dtype=torch.int32).contiguous()
    op(max_score_kv, nvb, out, init_blocks, local_blocks)
    return out


def register(model) -> int:
    """Register ``select`` on every layer's MSA indexer that has an unset hook; the number registered (0 when off)."""
    if not TSEL:
        logger.info(
            "[M3.1 dep2ctx] DEP2CTX_TSEL off (TRTLLM_M31_DISABLE_DEP2CTX_TSEL=1): built-in select"
        )
        return 0
    registered = 0
    for layer in model.model.layers:
        indexer = getattr(layer.self_attn.attn, "indexer", None)  # only the MSA backend has one
        if indexer is not None and getattr(indexer, "selector_hook", False) is None:
            indexer.selector_hook = select
            registered += 1
    return registered
