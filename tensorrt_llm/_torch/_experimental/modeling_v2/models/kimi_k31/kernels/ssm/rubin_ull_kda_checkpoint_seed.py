# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton kernel of the KDA checkpoint seed of transferred requests (``ops/ssm/
rubin_ull_kda_dsa_cache_manager.py``): every local KDA layer's checkpoint 0 and committed conv
window of the seeded slots, from the live pools, in one launch.

Per layer ``l`` and seeded slot ``s``: ``ckpt0[l, s] = ssm[l, s]`` (the state, flattened: unit
inner stride on both sides) and ``vconv[l, s, c, :window] = conv[l, s, c, :window]`` (rows ``c`` of
unit inner stride), each converted to the destination's dtype. Program ``(l, s, chunk)`` copies one
``BLOCK``-element chunk of the state or one row chunk of the conv window: pure copies (a dtype
conversion where the dtypes differ), so the bytes are those of ``index_select`` + ``index_copy_``.
"""

import triton
import triton.language as tl


@triton.jit
def _seed_checkpoints(
    slots_ptr,
    ssm_ptr,
    ssm_layer,
    ssm_slot,
    ckpt_ptr,
    ckpt_layer,
    ckpt_slot,
    conv_ptr,
    conv_layer,
    conv_slot,
    conv_row,
    vconv_ptr,
    vconv_layer,
    vconv_slot,
    vconv_row,
    state_elems,
    conv_rows,
    window,
    STATE_CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    layer = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + tl.program_id(1)).to(tl.int64)
    chunk = tl.program_id(2)
    # The branches name their pointers apart: Triton unifies a name's type across them.
    if chunk < STATE_CHUNKS:
        i = chunk * BLOCK + tl.arange(0, BLOCK)
        sm = i < state_elems
        state_src = ssm_ptr + layer * ssm_layer + slot * ssm_slot
        state_dst = ckpt_ptr + layer * ckpt_layer + slot * ckpt_slot
        state = tl.load(state_src + i, mask=sm)
        tl.store(state_dst + i, state.to(state_dst.dtype.element_ty), mask=sm)
    else:
        r = (chunk - STATE_CHUNKS) * ROWS + tl.arange(0, ROWS)
        c = tl.arange(0, BLOCK_W)
        cm = (r[:, None] < conv_rows) & (c[None, :] < window)
        conv_src = conv_ptr + layer * conv_layer + slot * conv_slot
        conv_dst = vconv_ptr + layer * vconv_layer + slot * vconv_slot
        rows = tl.load(conv_src + r[:, None] * conv_row + c[None, :], mask=cm)
        tl.store(
            conv_dst + r[:, None] * vconv_row + c[None, :],
            rows.to(conv_dst.dtype.element_ty),
            mask=cm,
        )


_BLOCK = 4096
_ROWS = 64


def seed_checkpoints(ssm, ckpt0, conv, vconv_window, slots) -> None:
    """``ckpt0[:, slots] = ssm[:, slots]`` and ``vconv_window[:, slots] = conv[:, slots]`` (dtype
    converted) in one launch. ``ssm`` / ``ckpt0`` are ``[layers, slots, *state]`` views whose state
    is contiguous; ``conv`` / ``vconv_window`` are ``[layers, slots, rows, window]`` views of unit
    inner stride; ``slots`` a 1-D int tensor on the device."""
    layers, n = ssm.shape[0], slots.numel()
    if n == 0 or layers == 0:
        return
    state_elems = ssm[0, 0].numel()
    conv_rows, window = conv.shape[-2], conv.shape[-1]
    state_chunks = triton.cdiv(state_elems, _BLOCK)
    conv_chunks = triton.cdiv(conv_rows, _ROWS)
    _seed_checkpoints[(layers, n, state_chunks + conv_chunks)](
        slots,
        ssm,
        ssm.stride(0),
        ssm.stride(1),
        ckpt0,
        ckpt0.stride(0),
        ckpt0.stride(1),
        conv,
        conv.stride(0),
        conv.stride(1),
        conv.stride(2),
        vconv_window,
        vconv_window.stride(0),
        vconv_window.stride(1),
        vconv_window.stride(2),
        state_elems,
        conv_rows,
        window,
        STATE_CHUNKS=state_chunks,
        BLOCK=_BLOCK,
        ROWS=_ROWS,
        BLOCK_W=triton.next_power_of_2(window),
        num_warps=4,
    )


def seed_checkpoints_applies(ssm, ckpt0, conv, vconv_window) -> bool:
    """Whether :func:`seed_checkpoints` covers these views: CUDA, the states contiguous within a
    slot on both sides, the conv rows of unit inner stride, matching shapes."""
    if not all(t.is_cuda for t in (ssm, ckpt0, conv, vconv_window)):
        return False
    if ssm.shape != ckpt0.shape or conv.shape != vconv_window.shape or conv.dim() != 4:
        return False
    for state in (ssm, ckpt0):
        inner = state[0, 0]
        if not inner.is_contiguous():
            return False
    return conv.stride(-1) == 1 and vconv_window.stride(-1) == 1
