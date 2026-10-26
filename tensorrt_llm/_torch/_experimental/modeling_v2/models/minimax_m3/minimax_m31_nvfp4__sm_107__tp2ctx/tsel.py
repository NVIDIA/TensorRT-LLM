# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_TSEL (#20629 TILE_SELECT, A2) on the sm_107 tp2ctx target: the prefill top-16 block select as the tile
select.

``minimax_m31_select_prefill_tile`` reads each 32-byte sector of the head-major proxy scores once for 8 queries of a KV
head; the built-in ``minimax_m3_select_blocks`` reads one score per sector per pass. Its table equals the built-in one
bit for bit (gate a: test_modeling_v2_m31_vr_tp2ctx_tsel_gpu.py on sm_107). Registered on every layer's MSA indexer
``selector_hook``; steps outside its rows / blocks range, graph captures and builds without the op keep the built-in
select.

Kill switch: ``TRTLLM_M31_DISABLE_TP2CTX_TSEL=1``. Gate a in-model (correctness runs only): ``TRTLLM_M31_CHECK_TSEL=1``
also runs ``minimax_m3_select_blocks`` on every tile-selected step and raises on any table difference.
"""

import os

import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    MSA_REQUIRED_TOPK,
)
from tensorrt_llm.logger import logger

from .m31_switches import m31_disabled

TSEL = not m31_disabled("TP2CTX_TSEL")
TSEL_MIN_ROWS = 256  # #20629's gate: below it the built-in one-CTA-per-row grid is at least as fast
TSEL_MAX_BLOCKS = 5856  # the op's shared-memory bound (kMinimaxM3SelectPrefillTileMaxBlocks)
TSEL_CHECK = os.environ.get("TRTLLM_M31_CHECK_TSEL") == "1"

_STATE: dict = {}


def _op():
    """``torch.ops.trtllm.minimax_m31_select_prefill_tile`` on sm_107, or None. Decided once per process."""
    if "op" not in _STATE:
        op = None
        if torch.cuda.get_device_capability() == (10, 7):
            op = getattr(torch.ops.trtllm, "minimax_m31_select_prefill_tile", None)
        _STATE["op"] = op
        logger.info(
            "[M3.1 tp2ctx] TP2CTX_TSEL prefill tile select: "
            f"{'engaged' if op is not None else 'not taken: not sm_107 or the op is not built'} "
            f"(rows >= {TSEL_MIN_ROWS}, blocks <= {TSEL_MAX_BLOCKS})"
        )
    return _STATE["op"]


def select(max_score_kv, n_valid_blocks, init_blocks, local_blocks):
    """The MSA indexer's selector hook: the head-major [total_q, num_kv_heads, 16] table of
    ``select_blocks_from_maxscore``, or None (outside the tile select's rows / blocks range, or no op) to keep it."""
    if (
        torch.cuda.is_current_stream_capturing()  # the host-side rows / blocks gate is not graph-stable
        or max_score_kv.dtype != torch.float32
        or max_score_kv.dim() != 3
        or max_score_kv.stride(2) != 1
        or max_score_kv.shape[2] < TSEL_MIN_ROWS
        or max_score_kv.shape[1] > TSEL_MAX_BLOCKS
    ):
        return None
    op = _op()
    if op is None:
        return None
    num_kv_heads, num_blocks, total_q = max_score_kv.shape
    out = torch.empty(
        (num_kv_heads, total_q, MSA_REQUIRED_TOPK), dtype=torch.int32, device=max_score_kv.device
    ).permute(1, 0, 2)
    nvb = n_valid_blocks.to(device=max_score_kv.device, dtype=torch.int32).contiguous()
    op(max_score_kv, nvb, out, init_blocks, local_blocks)
    if TSEL_CHECK:
        ref = torch.ops.trtllm.minimax_m3_select_blocks(
            max_score_kv, nvb, MSA_REQUIRED_TOPK, init_blocks, local_blocks, True
        )
        if not torch.equal(out, ref):
            raise AssertionError(
                f"TP2CTX_TSEL check: tile select table differs from minimax_m3_select_blocks "
                f"({int((out != ref).any(dim=-1).sum())} rows; {num_blocks} blocks, {total_q} rows)"
            )
        _STATE["checked"] = _STATE.get("checked", 0) + 1
        if _STATE["checked"] in (1, 100, 1000):
            logger.info(
                f"[M3.1 tp2ctx] TP2CTX_TSEL check: {_STATE['checked']} tile-selected steps equal"
            )
    return out


def register(model) -> int:
    """Register ``select`` on every layer's MSA indexer that has an unset hook; the number registered (0 when off)."""
    if not TSEL:
        logger.info(
            "[M3.1 tp2ctx] TP2CTX_TSEL off (TRTLLM_M31_DISABLE_TP2CTX_TSEL=1): built-in select"
        )
        return 0
    registered = 0
    for layer in model.model.layers:
        indexer = getattr(layer.self_attn.attn, "indexer", None)  # only the MSA backend has one
        if indexer is not None and getattr(indexer, "selector_hook", False) is None:
            indexer.selector_hook = select
            registered += 1
    return registered
