# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An MXFP8 x MXFP8 -> bf16 GEMM on TRT-LLM's SM107 persistent dense kernel (CuteDSLMXFP8RubinLinear) at a pinned
tactic, shared by the MiniMax-M3.1 targets. Each target keeps its own shape -> tactic table and switch.

The kernel compiles on its first eager call (a warmup, before capture). An [N, K] whose first call fails is declined
for the process, with a warning, and ``run`` returns None for it from then on, so the caller keeps its own path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm.logger import logger

_RUNNER = None
_DECLINED: set = set()
_LOGGED: set = set()


def _runner():
    global _RUNNER
    if _RUNNER is None:
        from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

        cls = getattr(cute_dsl_custom_ops, "CuteDSLMXFP8RubinLinear", None)
        _RUNNER = cls(output_dtype=torch.bfloat16) if cls is not None else False
        if cls is None:
            logger.info("[M3.1] CuteDSLMXFP8RubinLinear is unavailable")
    return _RUNNER or None


def declined(n: int, k: int) -> bool:
    """Whether an [N, K] failed its first call in this process."""
    return (n, k) in _DECLINED


def run(
    act_e4m3: torch.Tensor,
    act_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    tactic: Tuple,
    tag: str,
) -> Optional[torch.Tensor]:
    """[m, k] x [n, k]^T in bf16 at ``tactic``, or None (the runner unavailable, or the shape declined).

    ``tag`` prefixes the one taken line per (m, n, k) and the decline warning, e.g. ``[M3.1 DEP] DEP_GEMM_PROVIDER``.
    """
    m, (n, k) = int(act_e4m3.shape[0]), (int(weight.shape[0]), int(weight.shape[1]))
    if (n, k) in _DECLINED:
        return None
    runner = _runner()
    if runner is None:
        return None
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import _get_mxfp8_gemm_alpha

    capturing = torch.cuda.is_current_stream_capturing()
    try:
        output = runner(
            [act_e4m3, weight, act_sf, weight_scale, _get_mxfp8_gemm_alpha(act_e4m3.device)],
            tactic=tactic,
        )
    except Exception as err:  # noqa: BLE001 - a shape the kernel cannot run keeps the caller's path
        if capturing:
            raise
        _DECLINED.add((n, k))
        logger.warning(f"{tag} declined [{n}, {k}]: {type(err).__name__}: {err}")
        return None
    if not capturing and (m, n, k) not in _LOGGED:
        _LOGGED.add((m, n, k))
        logger.info(f"{tag} taken: [{m}, {k}] x [{n}, {k}]^T on SM107 split-K {tactic[-1]}")
    return output
