# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned FlashInfer CuTe DSL MXFP8 GEMM tactics for SM107 (MiniMax-M3 TP2 projections), above 32 rows.

TRT-LLM runs FlashInfer's CuTe DSL MXFP8 GEMM with FlashInfer's config sweep skipped, so above 32 rows (the
split-K low-M kernel's limit) every call takes the default tile (128, 128), cluster (1, 1): ceil(N / 128) CTAs, a
quarter of the SMs streaming the weight. This table holds FlashInfer's own tuned tactic per (N, K, M bucket) on
SM107. Every entry is non-split-K (split_k 1, each output keeps its full-K accumulation order), so its output equals
the default tactic's bit for bit (measured on all entries), at 18..40% less time from 33 to 16384 rows. A fixed
table rather than live tuning keeps the choice identical across runs and nodes. Used where TRT-LLM calls the
FlashInfer runner (decode-graph capture, autotune); eager prefill takes other GEMMs. Off unless armed: the executor
calls arm() for a model whose RuntimeHooks set mxfp8_pinned_tactics. Kill: TRTLLM_MXFP8_CUTE_PIN=0.
"""

import os
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

ENABLED = os.environ.get("TRTLLM_MXFP8_CUTE_PIN", "1") == "1"
MIN_ROWS = 33  # 1..32 rows keep FlashInfer's split-K low-M path (its tuned tactics there are split-K: not bitwise)
MAX_BUCKET = 16384

Tactic = Tuple[Tuple[int, int], Tuple[int, int], bool, bool, int]
# (N, K, M bucket = next power of two >= M, 64..16384)
#   -> (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch, split_k)
TACTICS: Dict[Tuple[int, int, int], Tactic] = {
    (3072, 6144, 64): ((256, 64), (2, 1), True, False, 1),
    (3072, 6144, 128): ((256, 64), (2, 1), True, False, 1),
    (3072, 6144, 256): ((256, 64), (2, 1), True, False, 1),
    (3072, 6144, 512): ((256, 64), (2, 1), False, False, 1),
    (3072, 6144, 1024): ((256, 128), (2, 1), False, False, 1),
    (3072, 6144, 2048): ((256, 256), (2, 1), False, False, 1),
    (3072, 6144, 4096): ((256, 192), (4, 2), False, False, 1),
    (3072, 6144, 8192): ((256, 192), (2, 1), False, False, 1),
    (3072, 6144, 16384): ((256, 192), (2, 1), True, False, 1),
    (4608, 6144, 64): ((256, 64), (2, 1), True, False, 1),
    (4608, 6144, 128): ((256, 64), (2, 1), True, False, 1),
    (4608, 6144, 256): ((256, 64), (2, 4), False, True, 1),
    (4608, 6144, 512): ((256, 128), (2, 2), False, False, 1),
    (4608, 6144, 1024): ((256, 192), (2, 1), False, False, 1),
    (4608, 6144, 2048): ((256, 192), (2, 1), False, False, 1),
    (4608, 6144, 4096): ((256, 256), (2, 1), False, False, 1),
    (4608, 6144, 8192): ((256, 192), (2, 4), True, False, 1),
    (4608, 6144, 16384): ((256, 256), (2, 1), True, False, 1),
    (4992, 6144, 64): ((256, 64), (2, 1), True, False, 1),
    (4992, 6144, 128): ((256, 64), (2, 1), True, False, 1),
    (4992, 6144, 256): ((256, 64), (2, 4), False, True, 1),
    (4992, 6144, 512): ((256, 128), (4, 2), False, True, 1),
    (4992, 6144, 1024): ((256, 192), (2, 1), False, False, 1),
    (4992, 6144, 2048): ((256, 192), (2, 1), False, False, 1),
    (4992, 6144, 4096): ((256, 192), (2, 1), False, False, 1),
    (4992, 6144, 8192): ((256, 192), (2, 1), False, False, 1),
    (4992, 6144, 16384): ((256, 192), (2, 1), True, False, 1),
    (6144, 1536, 64): ((256, 64), (2, 1), True, False, 1),
    (6144, 1536, 128): ((256, 64), (2, 1), True, False, 1),
    (6144, 1536, 256): ((256, 64), (2, 1), True, False, 1),
    (6144, 1536, 512): ((256, 128), (2, 1), False, False, 1),
    (6144, 1536, 1024): ((256, 256), (2, 1), False, False, 1),
    (6144, 1536, 2048): ((256, 192), (2, 4), False, False, 1),
    (6144, 1536, 4096): ((256, 192), (2, 1), False, False, 1),
    (6144, 1536, 8192): ((256, 192), (2, 1), False, False, 1),
    (6144, 1536, 16384): ((256, 192), (2, 1), True, False, 1),
    (6144, 4096, 64): ((256, 64), (2, 1), True, False, 1),
    (6144, 4096, 128): ((256, 64), (4, 1), True, False, 1),
    (6144, 4096, 256): ((256, 64), (2, 1), True, False, 1),
    (6144, 4096, 512): ((256, 128), (2, 1), False, False, 1),
    (6144, 4096, 1024): ((256, 256), (2, 1), False, False, 1),
    (6144, 4096, 2048): ((256, 192), (2, 4), False, False, 1),
    (6144, 4096, 4096): ((256, 192), (2, 1), False, False, 1),
    (6144, 4096, 8192): ((256, 192), (2, 1), True, False, 1),
    (6144, 4096, 16384): ((256, 192), (2, 1), True, False, 1),
    (6144, 6144, 64): ((256, 64), (2, 1), True, False, 1),
    (6144, 6144, 128): ((256, 64), (2, 1), True, False, 1),
    (6144, 6144, 256): ((256, 64), (2, 1), True, False, 1),
    (6144, 6144, 512): ((256, 128), (2, 1), True, False, 1),
    (6144, 6144, 1024): ((256, 256), (2, 1), False, False, 1),
    (6144, 6144, 2048): ((256, 192), (2, 4), False, False, 1),
    (6144, 6144, 4096): ((256, 192), (2, 1), False, False, 1),
    (6144, 6144, 8192): ((256, 192), (2, 1), False, False, 1),
    (6144, 6144, 16384): ((256, 256), (2, 1), True, False, 1),
    (12288, 6144, 64): ((256, 64), (2, 1), True, False, 1),
    (12288, 6144, 128): ((128, 64), (1, 2), True, False, 1),
    (12288, 6144, 256): ((256, 128), (2, 1), False, False, 1),
    (12288, 6144, 512): ((256, 256), (2, 1), False, False, 1),
    (12288, 6144, 1024): ((256, 192), (2, 2), False, False, 1),
    (12288, 6144, 2048): ((256, 192), (2, 1), False, False, 1),
    (12288, 6144, 4096): ((256, 192), (2, 1), False, False, 1),
    (12288, 6144, 8192): ((256, 256), (2, 1), False, False, 1),
    (12288, 6144, 16384): ((256, 256), (2, 1), True, False, 1),
}

_STATE: dict = {}
# Process-wide once armed: another model later or concurrently in the same process, at a table shape on SM107,
# also takes the pinned tactic. Its outputs are unchanged (each entry is bitwise the default tactic), only faster.
# The LLM API runs one executor per worker process.
_ARMED = [False]


def arm() -> None:
    """Engage the table for this process (a model opted in through RuntimeHooks.mxfp8_pinned_tactics)."""
    _ARMED[0] = True


def _bucket(m: int) -> int:
    return max(64, 1 << (m - 1).bit_length())


def tactic_for(m: int, n: int, k: int) -> Optional[Tactic]:
    """The pinned tactic of an [m, k] x [n, k]^T call, or None (FlashInfer's own path)."""
    if not ENABLED or not _ARMED[0] or m < MIN_ROWS or m > MAX_BUCKET or get_sm_version() != 107:
        return None
    return TACTICS.get((n, k, _bucket(m)))


def run(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """The MXFP8 GEMM act @ weight^T on FlashInfer's CuTe DSL runner at the pinned tactic, or None when no entry covers
    the call (the caller runs FlashInfer's default path). Bitwise equal to that default path where an entry exists."""
    m, k = act.shape
    n = weight.shape[0]
    tactic = tactic_for(m, n, k)
    if tactic is None:
        return None
    try:
        from flashinfer.gemm import gemm_base as _gb

        key = (out_dtype, act.device)
        st = _STATE.get(key)
        if st is None:
            major, minor = _gb.get_compute_capability(act.device)
            st = _STATE[key] = (
                _gb._cute_dsl_gemm_mxfp8_runner(major, minor, True, out_dtype),
                _gb._get_cache_buf("mm_mxfp8_workspace", _gb.DEFAULT_WORKSPACE_SIZE, act.device),
            )
            logger.info(
                "[MXFP8 GEMM] pinned CuTe DSL tactics engaged above 32 rows (TRTLLM_MXFP8_CUTE_PIN)"
            )
        runner, workspace = st
        out = torch.empty((m, n), dtype=out_dtype, device=act.device)
        runner(
            inputs=[act, weight.t(), act_scale, weight_scale, out_dtype, out, workspace],
            tactic=tactic,
        )
        return out
    except Exception as e:  # noqa: BLE001 - a FlashInfer API change falls back to the default path, once loudly
        if "failed" not in _STATE:
            _STATE["failed"] = True
            logger.warning(
                f"[MXFP8 GEMM] pinned CuTe DSL tactic unavailable ({type(e).__name__}: {e}); default path"
            )
        return None
