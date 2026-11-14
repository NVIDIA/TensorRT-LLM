# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The speculative step's store and next-step staging at the tail of the forward (one Triton launch).

For each of the first ``n`` rows (``n`` read from device memory, so a captured graph whose step was not armed writes
nothing), with ``s = slots[i]``:

* the spec sampler's per-slot store, exactly as ``m3_spec_tail.slot_scatter`` writes it (zero-padded to each store
  width): ``new[s][c]``, ``next_new[s][c]``, ``draft[s][c]`` and ``lens[s]``;
* the engine's next-step inputs, exactly as ``m3_spec_tail.step_gather`` writes them from that store for a step whose
  rows are these rows in this order (no new tokens, first previous row 0): ``ids[i*T + t] = next_new[s][t]``,
  ``draft_ids[i*D + d] = draft[s][d]``, ``pos_off[i*T + t] = lens[s]``, ``kv_off[i] = lens[s] - T``.

Integer copies only: the values are the ones the sampler's and the engine's own copies write.

With ``cpp`` the same writes run as this target's C++ op (trtllm::minimax_m3_sm107_tp8_fold_tail,
minimaxM3Sm107/foldTail.cu: one 32-lane CTA per row) instead of the Triton kernel, for widths up to 32.
"""

import torch
import triton as triton

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.spec.m3_fold_tail import (
    _fold_tail_kernel,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.spec.fold_tail import (
    fold_tail as _shared_fold_tail,
)


def _launch_cpp(*args: object, one_warp: bool) -> None:
    op = (
        torch.ops.trtllm.minimax_m3_sm107_tp8_fold_tail_one_warp
        if one_warp
        else torch.ops.trtllm.minimax_m3_sm107_tp8_fold_tail
    )
    op(*args)


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
    one_warp: bool = False,
) -> bool:
    """Launch over ``rows`` programs (the first ``ctrl[0]`` write); False, having launched nothing, when the inputs
    are outside what the kernel covers. Store views are 2-D [slots, width] (any strides), sources [rows, width].
    ``one_warp`` (with ``cpp``): the in-kernel form (one warp looping over the rows, foldTailWarp in
    minimaxM3Sm107/foldTail.h) that a caller folding the tail into its own kernel runs; the same writes."""
    return _shared_fold_tail(
        ctrl=ctrl,
        slots=slots,
        rows=rows,
        src_new=src_new,
        dst_new=dst_new,
        src_next=src_next,
        dst_next=dst_next,
        src_draft=src_draft,
        dst_draft=dst_draft,
        src_lens=src_lens,
        dst_lens=dst_lens,
        ids=ids,
        draft_ids=draft_ids,
        pos_off=pos_off,
        kv_off=kv_off,
        tokens_per_step=tokens_per_step,
        cpp=cpp,
        native_launch=lambda *args: _launch_cpp(*args, one_warp=one_warp),
        triton_kernel=_fold_tail_kernel,
    )
