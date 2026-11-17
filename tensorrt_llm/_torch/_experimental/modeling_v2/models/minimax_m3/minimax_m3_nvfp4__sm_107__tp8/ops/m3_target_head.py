# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target lm_head shard of a verify step of at most 8 rows, streamed from a lossless 12-bit coding of its BF16
weight (75% of the bytes; ``ops.gemm.m3_target_head`` has the format).

``prepare_head`` codes this rank's shard once, before the executor's warmup, and keeps it only when it decodes to the
BF16 weight bit for bit (``round_trip_ok``); a shard that does not keeps the in-tree BF16 GEMM. ``logits`` runs the
kernel: BF16 x BF16 MMAs with fp32 accumulation over the full K and one BF16 rounding per logit, the in-tree head's
arithmetic in another summation order. The kernel is compiled into TensorRT-LLM
(cpp/tensorrt_llm/kernels/minimaxM3/targetHead.cu) and called through the catalog entry ``ops.gemm.m3_target_head``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, cast

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    target_head_state as _head_shared,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm.m3_target_head import (
    CTA_WARPS as CTA_WARPS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm.m3_target_head import (
    MAX_ROWS as MAX_ROWS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm.m3_target_head import (
    encode_chunked as encode_chunked,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm.m3_target_head import (
    m3_target_head as _target_head_op,  # noqa: F401 — used by the bound family implementation
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm.m3_target_head import (
    round_trip_ok as round_trip_ok,
)


@dataclass(eq=False)
class TargetHead:
    """One rank's target lm_head shard in the 12-bit coding, and the kernel's split-tile scratch.

    ``coded`` is False when the coding did not reproduce the weight bit for bit (the in-tree GEMM then runs)."""

    weight: torch.Tensor
    coded: bool
    w: Optional[torch.Tensor] = None
    base: int = 0
    exc_off: Optional[torch.Tensor] = None
    exc_pos: Optional[torch.Tensor] = None
    exc_w: Optional[torch.Tensor] = None
    part: Optional[torch.Tensor] = None
    counter: Optional[torch.Tensor] = None
    grid: int = 0


_NUM_SMS: Optional[int] = None


def _num_sms() -> int:
    return _head_shared._num_sms(_HEAD_TARGET)


def head_supported(lm_head) -> bool:
    """A bias-free 2-D BF16 head of whole 16-row tiles whose K the coding covers (a multiple of 64)."""
    return _head_shared.head_supported(_HEAD_TARGET, lm_head)


def prepare_head(lm_head, grid: Optional[int] = None) -> TargetHead:
    """Code ``lm_head``'s local shard and check it decodes to the BF16 weight bit for bit (fails closed: a shard that
    does not is returned with ``coded=False`` and nothing kept). Built once, eagerly, before the executor's warmup
    (``M3Eagle3Worker.prebuild_target_head``)."""
    return _head_shared.prepare_head(_HEAD_TARGET, lm_head, grid)


def logits(head: TargetHead, x: torch.Tensor, use_pdl: bool = True) -> torch.Tensor:
    """This rank's BF16 logits shard [rows, n] of the rows x [rows <= MAX_ROWS, K] (BF16) from the coded shard."""
    return _head_shared.logits(_HEAD_TARGET, head, x, use_pdl)


_HEAD_TARGET = cast(_head_shared.HeadTarget, sys.modules[__name__])
