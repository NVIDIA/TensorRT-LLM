# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The draft_front contract: the MTP draft front feeding the ROW-parallel ``eh_proj``.

Native semantics (``DeepseekV3MTP.forward``), with ``H`` the hidden size::

    x = concat([enorm(embeds), hnorm(hidden)], dim=-1)  # [R, 2H]
    x = chunk(x, tp_size, dim=-1)[tp_rank]  # [R, 2H / tp_size] view
    out = eh_proj(x)  # ROW-parallel, all-reduced

``pack_draft_front`` returns the same ``eh_proj`` input. Only the source(s) that
the rank's chunk covers are normalized and only the chunk's columns are written.
The result is a view into a fresh ``[R, 2H]`` buffer with the native chunk's
shape, strides and offset, so the ``eh_proj`` GEMM sees an operand laid out
exactly like the native one. The other columns of the buffer are never read.

Embedding lookup, ``eh_proj`` and its reduction stay with the caller.

The kernel mirrors the FlashInfer CuTe DSL RMSNorm, so draft_front serves a call only when
the native ``enorm``/``hnorm`` would run that kernel. :func:`unsupported_reason`
checks the per-process conditions on every call; the caller checks the norm
modules once with :func:`~.native_norm.native_norm_reason`.
"""

from typing import NamedTuple

import torch

from ... import dispatch
from .native_norm import flashinfer_path_reason

_SUPPORTED_DTYPES = (torch.bfloat16,)
_VEC_ELEMS = 8  # 16-byte vectors of BF16
_MAX_HIDDEN_SIZE = 16384  # one CTA per row, as in the native norm kernel


class Segment(NamedTuple):
    """Columns ``[begin, end)`` of concat source ``source`` (0: embeds, 1: hidden)."""

    source: int
    begin: int
    end: int


def plan_segments(hidden_size: int, tp_size: int, tp_rank: int) -> tuple[Segment, ...]:
    """Map rank ``tp_rank``'s chunk of the ``[2H]`` concatenation onto its sources.

    ``tp_size == 1`` means the whole concatenation (the unsharded native input).

    Raises:
        ValueError: if ``2 * hidden_size`` is not divisible by ``tp_size`` or the
            rank is out of range.
    """
    total = 2 * hidden_size
    if tp_size < 1 or total % tp_size != 0:
        raise ValueError(f"2*hidden_size={total} is not divisible by tp_size={tp_size}")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank={tp_rank} out of range for tp_size={tp_size}")
    width = total // tp_size
    lo, hi = tp_rank * width, (tp_rank + 1) * width
    segments = []
    for source in (0, 1):
        src_lo, src_hi = source * hidden_size, (source + 1) * hidden_size
        begin, end = max(lo, src_lo), min(hi, src_hi)
        if begin < end:
            segments.append(Segment(source, begin - src_lo, end - src_lo))
    return tuple(segments)


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


def _row_operand_reason(name: str, t: torch.Tensor, rows: int, hidden_size: int) -> str | None:
    if t.dim() != 2 or t.shape[0] != rows or t.shape[1] != hidden_size:
        return f"{name} shape {tuple(t.shape)} != ({rows}, {hidden_size})"
    if t.stride(0) % _VEC_ELEMS != 0:
        return f"{name} row stride {t.stride(0)} not a multiple of {_VEC_ELEMS}"
    if t.stride(1) != 1:
        return f"{name} column stride {t.stride(1)} != 1"
    if not _aligned(t):
        return f"{name} not 16-byte aligned"
    return None


def unsupported_reason(
    embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    tp_size: int,
    tp_rank: int,
) -> str | None:
    """Return why the candidate cannot serve these operands, or ``None`` if it can.

    The checks are host-side only and launch nothing, so a caller can take the
    native path with no side effect from the candidate. They include whether a
    native BF16 ``RMSNorm`` on this device runs the FlashInfer CuTe DSL kernel
    that draft_front mirrors; the per-module conditions (gemma, cuda.tile, NVFP4) are the
    caller's, see :func:`~.native_norm.native_norm_reason`.
    """
    if embeds.dim() != 2:
        return f"embeds must be 2-D, got {embeds.dim()}-D"
    rows, hidden_size = embeds.shape
    dtype = embeds.dtype
    if dtype not in _SUPPORTED_DTYPES:
        return f"dtype {dtype} not supported"
    tensors = (embeds, hidden_states, enorm_weight, hnorm_weight)
    if any(t.dtype != dtype for t in tensors):
        return "operand dtypes differ"
    if any(not t.is_cuda or t.device != embeds.device for t in tensors):
        return "operands must share one CUDA device"
    if hidden_size % _VEC_ELEMS != 0 or hidden_size > _MAX_HIDDEN_SIZE:
        return f"hidden_size {hidden_size} outside the supported domain"
    if tp_size < 1 or (2 * hidden_size) % tp_size != 0 or not 0 <= tp_rank < tp_size:
        return f"tp_size={tp_size}, tp_rank={tp_rank} invalid for hidden_size {hidden_size}"
    if (2 * hidden_size // tp_size) % _VEC_ELEMS != 0:
        return f"chunk width {2 * hidden_size // tp_size} not a multiple of {_VEC_ELEMS}"
    for name, t in (("embeds", embeds), ("hidden_states", hidden_states)):
        reason = _row_operand_reason(name, t, rows, hidden_size)
        if reason is not None:
            return reason
    for name, w in (("enorm_weight", enorm_weight), ("hnorm_weight", hnorm_weight)):
        if w.dim() != 1 or w.shape[0] != hidden_size or not w.is_contiguous() or not _aligned(w):
            return f"{name} must be a contiguous, 16-byte aligned [{hidden_size}] tensor"
    return flashinfer_path_reason(embeds.device)


def _enable_pdl() -> bool:
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    return get_env_enable_pdl()


def pack_draft_front(
    embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    enorm_eps: float,
    hnorm_weight: torch.Tensor,
    hnorm_eps: float,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor | None:
    """Build the TP-local ``eh_proj`` input of the MTP draft layer.

    Args:
        embeds: ``[R, H]`` BF16 draft-token embeddings (native ``embed_tokens``
            output).
        hidden_states: ``[R, H]`` BF16 hidden states entering the draft layer.
        enorm_weight: ``[H]`` BF16 weight of ``enorm``.
        enorm_eps: epsilon of ``enorm``.
        hnorm_weight: ``[H]`` BF16 weight of ``hnorm``.
        hnorm_eps: epsilon of ``hnorm``.
        tp_rank: this rank's index in the ``eh_proj`` TP group.
        tp_size: size of that group; 1 returns the full concatenation.

    Returns:
        A ``[R, 2H / tp_size]`` view equal to native chunk ``tp_rank`` of
        ``concat([enorm(embeds), hnorm(hidden_states)], -1)``, with the native
        chunk's strides; or ``None`` when the operands are outside the supported
        domain, when the native norm on this device would not run the FlashInfer
        CuTe DSL kernel, or when the kernel was not compiled before CUDA-graph
        capture (nothing was launched; run the native path).
    """
    reason = unsupported_reason(embeds, hidden_states, enorm_weight, hnorm_weight, tp_size, tp_rank)
    if reason is not None:
        dispatch.log_capability("draft_front", "draft_front_call_fallback", False, reason)
        return None
    rows, hidden_size = embeds.shape
    kernel = None
    if rows > 0:
        from .kernels import get_compiled_kernel

        segments = tuple(
            (s.source, s.begin, s.end) for s in plan_segments(hidden_size, tp_size, tp_rank)
        )
        kernel = get_compiled_kernel(
            embeds.dtype,
            hidden_size,
            segments,
            _enable_pdl(),
            allow_compile=not torch.cuda.is_current_stream_capturing(),
        )
        if kernel is None:
            dispatch.log_capability(
                "draft_front",
                "draft_front_call_fallback",
                False,
                "kernel not compiled before CUDA graph capture",
            )
            return None
    dispatch.log_capability("draft_front", "draft_front_call", True)
    width = 2 * hidden_size // tp_size
    concat = torch.empty((rows, 2 * hidden_size), dtype=embeds.dtype, device=embeds.device)
    if kernel is not None:
        kernel(
            embeds,
            hidden_states,
            enorm_weight,
            hnorm_weight,
            concat[:, :hidden_size],
            concat[:, hidden_size:],
            rows,
            float(enorm_eps),
            float(hnorm_eps),
        )
    return concat[:, tp_rank * width : (tp_rank + 1) * width]


__all__ = ["Segment", "pack_draft_front", "plan_segments", "unsupported_reason"]
