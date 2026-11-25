# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_LOGITS_WIDEN: the logits widening of ``LogitsProcessor.forward`` (``lm_head(hidden).float()``) in one vectorized
launch, through the logits widen override (``modules/logits_processor.set_logits_widen_override``).
The launch is the family CUDA C++ op ``mv2_minimax_m3::minimax_m31_widen_rows`` (``minimax_m3/ops/row_ops``; 256 threads
x 2 iterations of 16-byte loads and streaming stores per CTA).

The padded lm_head's output is sliced to the vocabulary (``output[..., :-padding_size]``, a strided view), and the stock
``.float()`` of that view runs PyTorch's strided copy, once for the target's verify rows and once for the drafter's
rows per step. ``widen`` reads each row's bf16 / fp16 values with 16-byte accesses and writes the contiguous float32
[rows, vocab] that ``.float()`` returns; widening is exact, so the logits are the stock ones bit for bit.

Contract: a 2-D bf16 / fp16 CUDA tensor with a unit inner stride, at least one element and at most 65535 rows, on a
build that carries the op; anything else returns None (logging a "not taken" line once per reason), and the stock
``.float()`` runs. Kill switch:
``TRTLLM_M31_DISABLE_DEP_LOGITS_WIDEN=1`` installs no override. Read at import.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm.logger import logger

from ..ops import row_ops
from .m31_switches import m31_disabled

DEP_LOGITS_WIDEN = not m31_disabled("DEP_LOGITS_WIDEN")

_DTYPES = (torch.bfloat16, torch.float16)
_TAKEN: set = set()
_NOT_TAKEN: set = set()


def outside_contract(logits: torch.Tensor) -> Optional[str]:
    """Why ``widen`` does not take ``logits``, or None (see the module docstring)."""
    if not isinstance(logits, torch.Tensor):
        return "not a tensor"
    if not logits.is_cuda or logits.dtype not in _DTYPES or logits.dim() != 2:
        return f"logits {logits.dtype} {tuple(logits.shape)} on {logits.device}"
    if logits.stride(1) != 1 or logits.numel() == 0 or logits.shape[0] > 65535:
        return f"logits {tuple(logits.shape)} strides {logits.stride()}"
    if row_ops.widen_rows_op() is None:
        return "mv2_minimax_m3::minimax_m31_widen_rows is not built"
    return None


def widen(logits: torch.Tensor) -> Optional[torch.Tensor]:
    """``logits.float()`` bit for bit (a contiguous float32 [rows, vocab]), or None."""
    reason = outside_contract(logits)
    if reason is not None:
        if reason not in _NOT_TAKEN:
            _NOT_TAKEN.add(reason)
            logger.info(
                f"[M3.1 DEP] DEP_LOGITS_WIDEN not taken ({reason}); the stock .float() runs"
            )
        return None
    rows, cols = logits.shape
    out = row_ops.widen_rows(logits)
    if logits.dtype not in _TAKEN:
        _TAKEN.add(logits.dtype)
        logger.info(
            f"[M3.1 DEP] DEP_LOGITS_WIDEN taken ({logits.dtype} at {rows} x {cols}, row stride {logits.stride(0)})"
        )
    return out


def engagement() -> str:
    """The log line's state: what the override does in this process."""
    if not DEP_LOGITS_WIDEN:
        return "off (TRTLLM_M31_DISABLE_DEP_LOGITS_WIDEN=1); the stock .float()"
    if row_ops.widen_rows_op() is None:
        return (
            "engaged, but mv2_minimax_m3::minimax_m31_widen_rows is not built; the stock .float()"
        )
    return "engaged (one vectorized CUDA C++ launch for the logits widening)"


def log_engagement() -> None:
    logger.info(f"[M3.1 DEP] DEP_LOGITS_WIDEN: {engagement()}")
