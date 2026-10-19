# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small fused kernels of the MiniMax-M3 Eagle3 step (GB300, decode).

``chunk_norm`` / ``capture_norm`` apply the drafter's per-layer fc_norm RMSNorms to the captured target hidden states
(the in-tree ``apply_eagle3_fc`` norms, same fp32 arithmetic); ``draft_input`` is a fused draft pass's input (the
embedding lookup, the drafter layer's input_layernorm / hidden_norm and their concat, same arithmetic);
``step_gather`` / ``slot_scatter`` move the step's integer outputs between the spec sampler's store and the engine's
input buffers.
"""

from typing import Optional, Tuple

import torch

# The kernels are compiled into TensorRT-LLM (cpp/tensorrt_llm/kernels/minimaxM3/specTail.cu) and reached through the
# ops/spec entries.
from ..ops.spec.m3_capture_norm import m3_capture_norm as _capture_norm_op
from ..ops.spec.m3_chunk_norm import m3_chunk_norm as _chunk_norm_op
from ..ops.spec.m3_draft_input import m3_draft_input as _draft_input_op
from ..ops.spec.m3_slot_scatter import m3_slot_scatter as _slot_scatter_op
from ..ops.spec.m3_step_gather import m3_step_gather as _step_gather_op


def draft_input(
    ids: torch.Tensor,
    table: torch.Tensor,
    hidden: torch.Tensor,
    rows: Optional[torch.Tensor],
    w_embed: torch.Tensor,
    eps_embed: float,
    w_hidden: torch.Tensor,
    eps_hidden: float,
    use_pdl: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(cat([rmsnorm(table[ids]) * w_embed, rmsnorm(hidden[rows]) * w_hidden], -1) [M, 2H], hidden[rows] [M, H]) for
    int32 ids [M] and int64 rows [M] (rows None: hidden[:M]); ids outside the table embed to zero."""
    m, h = ids.shape[0], table.shape[1]
    out = torch.empty((m, 2 * h), dtype=table.dtype, device=table.device)
    residual = torch.empty((m, h), dtype=table.dtype, device=table.device)
    return _draft_input_op(
        ids, table, hidden, rows, w_embed, w_hidden, eps_embed, eps_hidden, out, residual, use_pdl
    )


def chunk_norm(
    src: torch.Tensor, weights: torch.Tensor, eps: float, use_pdl: bool = True
) -> torch.Tensor:
    """Per-chunk RMSNorm of src [M, >= C*H] (row stride may exceed C*H) with weights [C, H] -> [M, C*H]."""
    c, h = weights.shape
    out = torch.empty((src.shape[0], c * h), dtype=src.dtype, device=src.device)
    return _chunk_norm_op(src, weights, eps, out, use_pdl)


# The overlap scheduler's per-step gathers and the spec sampler's slot stores as one kernel each (step_gather /
# slot_scatter) instead of ~10 and 4 small torch ops; called by the target's SpecStepFold.


def step_gather(
    new_tok, next_draft, lens, slots, ppi, P, T, D, ids, ids0, draft, d0, pos_off, kv_off, b0
) -> bool:
    """See step_gather_kernel; False when the inputs are outside what the kernel covers (caller falls back)."""
    ts = (new_tok, next_draft, lens, slots, ppi, ids, draft, pos_off, kv_off)
    if (
        any(t.dtype != torch.int32 or not t.is_cuda for t in ts)
        or next_draft.dim() != 2
        or new_tok.dim() < 2
    ):
        return False
    if (new_tok.dim() == 3 and new_tok.shape[2] != 1) or not (
        lens.is_contiguous()
        and ids.is_contiguous()
        and draft.is_contiguous()
        and pos_off.is_contiguous()
        and kv_off.is_contiguous()
    ):
        return False
    _step_gather_op(
        new_tok, next_draft, lens, slots, ppi, P, T, D, ids, ids0, draft, d0, pos_off, kv_off, b0
    )
    return True


def slot_scatter(
    slots, src_new, dst_new, src_next_new, dst_next_new, src_lens, dst_lens, src_draft, dst_draft
) -> bool:
    """See slot_scatter_kernel (2-D int32 views, zero padding); False to fall back."""
    ts = (src_new, dst_new, src_next_new, dst_next_new, src_lens, dst_lens, src_draft, dst_draft)
    if any(t.dtype != torch.int32 or not t.is_cuda for t in ts) or slots.dtype != torch.long:
        return False
    two = (src_new, dst_new, src_next_new, dst_next_new, src_draft, dst_draft)
    if any(t.dim() != 2 for t in two) or src_lens.dim() != 1 or dst_lens.dim() != 1:
        return False
    _slot_scatter_op(
        slots,
        src_new,
        dst_new,
        src_next_new,
        dst_next_new,
        src_lens,
        dst_lens,
        src_draft,
        dst_draft,
    )
    return True


def capture_norm(
    h: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    dst: torch.Tensor,
) -> None:
    """dst [M, H] (row stride may exceed H) = rmsnorm(h + residual) * weight with chunk_norm's rounding."""
    _capture_norm_op(h, residual, weight, eps, dst)
