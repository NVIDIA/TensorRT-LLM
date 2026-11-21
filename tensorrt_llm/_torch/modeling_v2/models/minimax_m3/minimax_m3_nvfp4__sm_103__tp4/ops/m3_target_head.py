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

from dataclasses import dataclass
from typing import Optional

import torch

from ..ops.gemm.m3_target_head import CTA_WARPS, MAX_ROWS, encode_chunked, round_trip_ok
from ..ops.gemm.m3_target_head import m3_target_head as _target_head_op


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
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    return _NUM_SMS


def head_supported(lm_head) -> bool:
    """A bias-free 2-D BF16 head of whole 16-row tiles whose K the coding covers (a multiple of 64)."""
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        return False
    return (
        getattr(lm_head, "bias", None) is None
        and weight.shape[0] % 16 == 0
        and weight.shape[1] % 64 == 0
    )


def prepare_head(lm_head, grid: Optional[int] = None) -> TargetHead:
    """Code ``lm_head``'s local shard and check it decodes to the BF16 weight bit for bit (fails closed: a shard that
    does not is returned with ``coded=False`` and nothing kept). Built once, eagerly, before the executor's warmup
    (``M3Eagle3Worker.prebuild_target_head``)."""
    weight = lm_head.weight
    with torch.no_grad():
        w = weight.detach()
        packed, base, exc_off, exc_pos, exc_w = encode_chunked(w)
        if not round_trip_ok(packed, base, exc_off, exc_pos, exc_w, w):
            return TargetHead(weight=weight, coded=False)
    if grid is None:
        grid = _num_sms()
    n = weight.shape[0]
    return TargetHead(
        weight=weight,
        coded=True,
        w=packed,
        base=base,
        exc_off=exc_off,
        exc_pos=exc_pos,
        exc_w=exc_w.view(-1),
        part=torch.zeros(grid * CTA_WARPS * 2 * 4 * 32, dtype=torch.float32, device=weight.device),
        counter=torch.zeros((n + 15) // 16, dtype=torch.int32, device=weight.device),
        grid=grid,
    )


def logits(head: TargetHead, x: torch.Tensor, use_pdl: bool = True) -> torch.Tensor:
    """This rank's BF16 logits shard [rows, n] of the rows x [rows <= MAX_ROWS, K] (BF16) from the coded shard."""
    assert head.coded and 1 <= x.shape[0] <= MAX_ROWS, (head.coded, x.shape)
    if x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
        x = x.contiguous()
    n = head.weight.shape[0]
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    return _target_head_op(
        head.w,
        x,
        out,
        head.part,
        head.counter,
        head.exc_off,
        head.exc_pos,
        head.exc_w,
        head.base,
        n,
        head.grid,
        use_pdl,
    )
