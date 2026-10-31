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

With ``cpp`` the same writes run as this target's C++ op (trtllm::minimax_m3_sm107_tp2_fold_tail,
minimaxM3Sm107Tp2/foldTail.cu: one 32-lane CTA per row) instead of the Triton kernel, for widths up to 32.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fold_tail_kernel(
    ctrl,
    slots,
    src_new,
    sn_s0,
    sn_s1,
    SN_W: tl.constexpr,
    dst_new,
    dn_s0,
    dn_s1,
    DN_W: tl.constexpr,
    src_next,
    sx_s0,
    sx_s1,
    SX_W: tl.constexpr,
    dst_next,
    dx_s0,
    dx_s1,
    DX_W: tl.constexpr,
    src_draft,
    sd_s0,
    sd_s1,
    SD_W: tl.constexpr,
    dst_draft,
    dd_s0,
    dd_s1,
    DD_W: tl.constexpr,
    src_lens,
    dst_lens,
    ids,
    draft_ids,
    pos_off,
    kv_off,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    n = tl.load(ctrl)
    if i < n:
        s = tl.load(slots + i).to(tl.int64)
        c = tl.arange(0, BLOCK)
        new = tl.load(src_new + i * sn_s0 + c * sn_s1, mask=c < SN_W, other=0)
        tl.store(dst_new + s * dn_s0 + c * dn_s1, new, mask=c < DN_W)
        nxt = tl.load(src_next + i * sx_s0 + c * sx_s1, mask=c < SX_W, other=0)
        tl.store(dst_next + s * dx_s0 + c * dx_s1, nxt, mask=c < DX_W)
        drf = tl.load(src_draft + i * sd_s0 + c * sd_s1, mask=c < SD_W, other=0)
        tl.store(dst_draft + s * dd_s0 + c * dd_s1, drf, mask=c < DD_W)
        ln = tl.load(src_lens + i)
        tl.store(dst_lens + s, ln)
        # The next step's inputs read the store's padded rows: zero past each source width.
        tl.store(ids + i * T + c, tl.where(c < DX_W, nxt, 0), mask=c < T)
        tl.store(draft_ids + i * D + c, tl.where(c < DD_W, drf, 0), mask=c < D)
        tl.store(pos_off + i * T + c, tl.zeros([BLOCK], dtype=tl.int32) + ln, mask=c < T)
        tl.store(kv_off + i, ln - T)


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
) -> bool:
    """Launch over ``rows`` programs (the first ``ctrl[0]`` write); False, having launched nothing, when the inputs
    are outside what the kernel covers. Store views are 2-D [slots, width] (any strides), sources [rows, width]."""
    two = (src_new, dst_new, src_next, dst_next, src_draft, dst_draft)
    one = (src_lens, dst_lens, ids, draft_ids, pos_off, kv_off)
    if any(t.dim() != 2 for t in two) or any(t.dim() != 1 or t.stride(0) != 1 for t in one):
        return False
    if (
        any(t.dtype != torch.int32 or not t.is_cuda for t in two + one + (ctrl,))
        or slots.dtype != torch.long
    ):
        return False
    t_step = tokens_per_step
    d_step = t_step - 1
    if (
        rows <= 0
        or slots.numel() < rows
        or any(t.shape[0] < rows for t in (src_new, src_next, src_draft, src_lens))
        or ids.numel() < rows * t_step
        or pos_off.numel() < rows * t_step
        or draft_ids.numel() < rows * d_step
        or kv_off.numel() < rows
    ):
        return False
    widths = (src_new.shape[1], dst_new.shape[1], src_next.shape[1], dst_next.shape[1])
    widths += (src_draft.shape[1], dst_draft.shape[1], t_step)
    if cpp and max(widths) <= 32:
        torch.ops.trtllm.minimax_m3_sm107_tp2_fold_tail(
            ctrl, slots, rows, src_new, dst_new, src_next, dst_next, src_draft, dst_draft,
            src_lens, dst_lens, ids, draft_ids, pos_off, kv_off, t_step,
        )  # fmt: skip
        return True
    block = triton.next_power_of_2(max(widths))
    _fold_tail_kernel[(rows,)](
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
