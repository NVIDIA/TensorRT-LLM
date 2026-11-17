# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared integer store/staging kernel; targets own launch policy."""

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
