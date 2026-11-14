# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_PAD_FILL: the spec sampler's CUDA-graph padding rows filled in place, writing only those rows, through the
padding-row fill override (``speculative/interface.set_padding_rows_fill_override``).

Each rejection-sampling call rewrites the padding requests' rows twice: ``SpecWorkerBase._zero_padding_rows`` gives them
one-hot logits (an out-of-place ``masked_fill`` over every row of the logits), and ``_uniform_padding_probs`` gives
them uniform probs after the filters (a ``masked_fill_`` over every row of the probs). ``pad_rows_fill`` writes the
padding rows only: the CTAs of a real row read the row's mask byte and store nothing. It fills the probs in place, and
a copy of the logits (the sampler's ``clone``, so the caller's logits stay unchanged). The padding rows get the stock
fills' values bit for bit and the real rows are untouched, so the sampled tokens and the stored proposals are the stock
ones. The fill is the family CUDA C++ op ``mv2_minimax_m3::minimax_m31_fill_padding_rows``
(``minimax_m3/ops/row_ops``): up to 2048 rows of whole 16-byte vectors take one CTA wave that lists the padded rows
from the mask, so the launch does not scale with the real rows; other rows take a per-row grid.

Contract: a contiguous 2-D float32 / bfloat16 / float16 CUDA ``rows`` and a contiguous bool ``is_padding`` with one
entry per row on the same device, on a build that carries the op. Anything else returns None, and the stock fills
(``masked_fill`` and the column-0 write) run. Kill switch:
``TRTLLM_M31_DISABLE_DEP_PAD_FILL=1`` installs no override. Read at import.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm.logger import logger

from ..ops import row_ops
from .m31_switches import m31_disabled

DEP_PAD_FILL = not m31_disabled("DEP_PAD_FILL")

_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def outside_contract(rows: torch.Tensor, is_padding: torch.Tensor) -> bool:
    """Whether ``rows`` / ``is_padding`` fall outside the kernel's contract (see the module docstring)."""
    return (
        not isinstance(rows, torch.Tensor)
        or not isinstance(is_padding, torch.Tensor)
        or not rows.is_cuda
        or rows.dim() != 2
        or rows.dtype not in _DTYPES
        or not rows.is_contiguous()
        or is_padding.dtype != torch.bool
        or is_padding.dim() != 1
        or is_padding.numel() != rows.shape[0]
        or not is_padding.is_contiguous()
        or is_padding.device != rows.device
        or rows.shape[0] > 65535
        or row_ops.fill_padding_rows_op() is None
    )


def pad_rows_fill(
    rows: torch.Tensor, is_padding: torch.Tensor, value: float, col0_value: Optional[float] = None
) -> Optional[torch.Tensor]:
    """The padding-row fill override: ``rows`` with each padding row set to ``value`` (column 0 to ``col0_value`` when
    given) in place, or None (the stock fills) outside the contract."""
    if outside_contract(rows, is_padding):
        return None
    row_ops.fill_padding_rows(rows, is_padding, value, col0_value)
    return rows


def engagement() -> str:
    """The log line's state: what the override does in this process."""
    if not DEP_PAD_FILL:
        return "off (TRTLLM_M31_DISABLE_DEP_PAD_FILL=1); stock full-vocab masked fills"
    if row_ops.fill_padding_rows_op() is None:
        return "engaged, but mv2_minimax_m3::minimax_m31_fill_padding_rows is not built; stock full-vocab masked fills"
    return (
        "engaged (CUDA-graph padding rows filled in place; the real rows are not read or written)"
    )


def log_engagement() -> None:
    logger.info(f"[M3.1 DEP] DEP_PAD_FILL: {engagement()}")
