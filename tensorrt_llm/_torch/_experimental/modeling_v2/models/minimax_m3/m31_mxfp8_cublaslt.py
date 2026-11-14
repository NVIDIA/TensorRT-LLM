# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An MXFP8 x MXFP8 -> bf16 GEMM through cuBLASLt at a pinned algorithm
(``mv2_minimax_m3::minimax_m31_mxfp8_cublaslt_gemm``, the nine CUBLASLT_ALGO_CONFIG attributes), shared by the
MiniMax-M3.1 targets. Each target keeps its own shape -> algorithm table and switch.

An [N, K] whose first call fails (the algorithm not valid for the shape on this build) is declined for the process,
with a warning, and ``run`` returns None for it from then on, so the caller keeps its own path.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm.logger import logger

_OP_NAME = "minimax_m31_mxfp8_cublaslt_gemm"
_DECLINED: set = set()
_LOGGED: set = set()


def op():
    """``family_ops().minimax_m31_mxfp8_cublaslt_gemm`` when this build carries it, else None."""
    return getattr(family_ops(), _OP_NAME, None)


def declined(n: int, k: int) -> bool:
    """Whether an [N, K] failed its first call in this process."""
    return (n, k) in _DECLINED


def run(
    act_e4m3: torch.Tensor,
    act_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    algo: Sequence[int],
    tag: str,
) -> Optional[torch.Tensor]:
    """[m, k] x [n, k]^T in bf16 at the pinned ``algo``, or None (the op unavailable, or the shape declined).

    ``tag`` prefixes the one taken line per (m, n, k) and the decline warning.
    """
    m, (n, k) = int(act_e4m3.shape[0]), (int(weight.shape[0]), int(weight.shape[1]))
    gemm = op()
    if gemm is None or (n, k) in _DECLINED:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    try:
        output = gemm(act_e4m3, act_sf, weight, weight_scale, list(algo))
    except Exception as err:  # noqa: BLE001 - an algorithm the shape cannot run keeps the caller's path
        if capturing:
            raise
        _DECLINED.add((n, k))
        logger.warning(f"{tag} declined [{n}, {k}]: {type(err).__name__}: {err}")
        return None
    if not capturing and (m, n, k) not in _LOGGED:
        _LOGGED.add((m, n, k))
        logger.info(f"{tag} taken: [{m}, {k}] x [{n}, {k}]^T on cuBLASLt algo {list(algo)}")
    return output
