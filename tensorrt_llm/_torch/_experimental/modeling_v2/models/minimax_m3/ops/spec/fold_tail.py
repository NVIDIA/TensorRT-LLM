# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared speculative tail guards and launch; targets bind their original native entry."""

from typing import Any, Callable

import torch
import triton


def fold_tail(
    ctrl: torch.Tensor,
    slots: torch.Tensor,
    rows: int,
    src_new: torch.Tensor,
    dst_new: torch.Tensor,
    src_next: torch.Tensor,
    dst_next: torch.Tensor,
    src_draft: torch.Tensor,
    dst_draft: torch.Tensor,
    src_lens: torch.Tensor,
    dst_lens: torch.Tensor,
    ids: torch.Tensor,
    draft_ids: torch.Tensor,
    pos_off: torch.Tensor,
    kv_off: torch.Tensor,
    tokens_per_step: int,
    cpp: bool = False,
    *,
    native_launch: Callable[..., None],
    triton_kernel: Any,
) -> bool:
    """Launch over ``rows`` programs (the first ``ctrl[0]`` write); False, having launched nothing, when the inputs
    are outside what the kernel covers. Store views are 2-D [slots, width] (any strides), sources [rows, width]."""
    two = (src_new, dst_new, src_next, dst_next, src_draft, dst_draft)
    one = (src_lens, dst_lens, ids, draft_ids, pos_off, kv_off)
    if any((t.dim() != 2 for t in two)) or any((t.dim() != 1 or t.stride(0) != 1 for t in one)):
        return False
    if (
        any((t.dtype != torch.int32 or not t.is_cuda for t in two + one + (ctrl,)))
        or slots.dtype != torch.long
    ):
        return False
    t_step = tokens_per_step
    d_step = t_step - 1
    if (
        rows <= 0
        or slots.numel() < rows
        or any((t.shape[0] < rows for t in (src_new, src_next, src_draft, src_lens)))
        or (ids.numel() < rows * t_step)
        or (pos_off.numel() < rows * t_step)
        or (draft_ids.numel() < rows * d_step)
        or (kv_off.numel() < rows)
    ):
        return False
    widths = (src_new.shape[1], dst_new.shape[1], src_next.shape[1], dst_next.shape[1])
    widths += (src_draft.shape[1], dst_draft.shape[1], t_step)
    if cpp and max(widths) <= 32:
        native_launch(
            ctrl,
            slots,
            rows,
            src_new,
            dst_new,
            src_next,
            dst_next,
            src_draft,
            dst_draft,
            src_lens,
            dst_lens,
            ids,
            draft_ids,
            pos_off,
            kv_off,
            t_step,
        )
        return True
    block = triton.next_power_of_2(max(widths))
    triton_kernel[rows,](
        ctrl,
        slots,
        src_new,
        src_new.stride(0),
        src_new.stride(1),
        src_new.shape[1],
        dst_new,
        dst_new.stride(0),
        dst_new.stride(1),
        dst_new.shape[1],
        src_next,
        src_next.stride(0),
        src_next.stride(1),
        src_next.shape[1],
        dst_next,
        dst_next.stride(0),
        dst_next.stride(1),
        dst_next.shape[1],
        src_draft,
        src_draft.stride(0),
        src_draft.stride(1),
        src_draft.shape[1],
        dst_draft,
        dst_draft.stride(0),
        dst_draft.stride(1),
        dst_draft.shape[1],
        src_lens,
        dst_lens,
        ids,
        draft_ids,
        pos_off,
        kv_off,
        t_step,
        d_step,
        block,
        num_warps=1,
    )
    return True
