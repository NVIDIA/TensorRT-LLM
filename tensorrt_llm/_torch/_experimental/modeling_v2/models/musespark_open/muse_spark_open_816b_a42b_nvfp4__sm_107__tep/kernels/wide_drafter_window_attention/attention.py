# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This target's drafter window attention on decode steps of 33..1024 rows (``wide_drafter_window_attention``).

``drafter_block_decode``'s written-out block decode (``windowed_block_decode``)
calls ``ops/spec/dflash_window_attention`` once per drafter layer, whose split
geometry is the one-request tuning (65 one-tile splits per request and KV head).
With this fast path installed, a block decode of ``MIN_ROWS``..``MAX_ROWS`` rows
(5 to 128 requests at K = 7) calls ``ops/spec/dflash_window_attention_wide``
instead: the same attention and the same fused append, cut into a few pipelined
splits per request. Block decodes of fewer rows (one request, and the two to
four requests of the 9..32-row band) keep the one-request op, its geometry and its
PDL merge trigger.

The install binds a ``WideWindowAttention`` to the drafter
(``draft_model._wide_window_attention``), which the block decode asks per call
(``takes``): a pure function of the call's row count, the same on every rank.
It needs ``drafter_block_decode``'s block decode bound to the drafter, and every
drafter layer's attention geometry one the wide op takes.
"""

from __future__ import annotations

import types
from typing import Optional

import torch
from torch import nn

from tensorrt_llm.logger import logger

from ...ops.spec.dflash_window_attention_wide import (
    dflash_window_attention_wide,
    wide_geometry_unsupported_reason,
)

#: Fewest rows (block rows x requests) a block decode sends to the wide op: five
#: requests at K = 7. Fewer rows are the one-request path and the 9..32-row band.
MIN_ROWS = 33
#: Most rows: 128 requests at K = 7, the widest step the op is certified on.
MAX_ROWS = 1024
#: The drafter attribute the block decode reads.
ATTRIBUTE = "_wide_window_attention"


class WideWindowAttention:
    """The wide window attention, for block decodes of ``MIN_ROWS``..``MAX_ROWS`` rows."""

    def takes(self, rows: int) -> bool:
        """Whether a block decode of ``rows`` rows calls this instead of the one-request op."""
        return MIN_ROWS <= rows <= MAX_ROWS

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_batch_idx: torch.Tensor,
        window_size: tuple[int, int],
        enable_pdl: bool,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        page_table: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """``dflash_window_attention_wide`` with the block decode's arguments (default scale)."""
        return dflash_window_attention_wide(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            cache_batch_idx,
            window_size,
            None,
            enable_pdl,
            k_new,
            v_new,
            page_table,
        )


def unsupported_reason(draft_model: Optional[nn.Module]) -> Optional[str]:
    """Why the wide window attention cannot be installed on ``draft_model``, or None."""
    from ..drafter_block_decode.block_decode import windowed_block_decode

    if draft_model is None:
        return "no drafter"
    bound = draft_model.__dict__.get("dflash_forward")
    if not (isinstance(bound, types.MethodType) and bound.__func__ is windowed_block_decode):
        return "the drafter does not run drafter_block_decode's block decode"
    block_size = getattr(draft_model, "block_size", None)
    if not block_size:
        return "the drafter declares no block size"
    reason = wide_geometry_unsupported_reason(int(block_size))
    if reason is not None:
        return reason
    return None


def install(draft_model: Optional[nn.Module]) -> bool:
    """Bind the wide window attention to the drafter's block decode. Idempotent.

    Call after ``drafter_block_decode``'s install. Returns whether this call
    changed the drafter.

    Raises:
        ValueError: when the drafter cannot take it (``unsupported_reason``).
    """
    reason = unsupported_reason(draft_model)
    if reason is not None:
        raise ValueError(
            f"[musespark_open] wide_drafter_window_attention cannot run here: {reason}"
        )
    if isinstance(draft_model.__dict__.get(ATTRIBUTE), WideWindowAttention):
        return False
    setattr(draft_model, ATTRIBUTE, WideWindowAttention())
    logger.info(
        "[musespark_open] wide_drafter_window_attention: drafter window attention of "
        f"{MIN_ROWS}..{MAX_ROWS}-row block decodes on the multi-request split geometry"
    )
    return True
