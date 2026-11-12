# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton helpers that reproduce the C++ ``causal_conv1d_fwd_kernel``'s arithmetic bit
for bit, for the one-pass GDN prefill front end.

* ``_fma_rn``: a single fp32 ``fma.rn`` (no contraction, no ftz), so the conv is the
  C++ kernel's ``__fmaf_rn`` chain from bias (or +0);
* ``_silu_ref``: ``v * rcp.rn(1 + ex2.approx((-v) * log2e))``, the non-ftz expansion of
  ``v * __frcp_rn(1.0f + __expf(-v))`` (the C++ op is built without fast math);
* ``_first_tile_tap``: ``x[t - K]`` in a sequence's first token tile -- the input, or
  the slot's initial conv state (zero without one);
* ``_final_state_tap``: column ``W`` of the final conv state, from the input tail (with
  the ``seqlen < width - 1`` shift of the old state).
"""

import triton
import triton.language as tl


@triton.jit
def _silu_ref(v):
    # Bitwise twin of causal_conv1d_fwd_kernel's v * __frcp_rn(1.0f + __expf(-v))
    # compiled without ftz: __expf(x) = ex2.approx(x * log2e) in libdevice.
    return tl.inline_asm_elementwise(
        "{\n"
        ".reg .f32 t;\n"
        "neg.f32 t, $1;\n"
        "mul.rn.f32 t, t, 0f3FB8AA3B;\n"
        "ex2.approx.f32 t, t;\n"
        "add.rn.f32 t, t, 0f3F800000;\n"
        "rcp.rn.f32 t, t;\n"
        "mul.rn.f32 $0, $1, t;\n"
        "}",
        "=f,f",
        [v],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fma_rn(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _first_tile_tap(
    x_base,
    x_offs,
    s_base,
    s_offs_c,
    ii,
    mask,
    has_init,
    stride_x_t,
    stride_s_k,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """x[t - K] in a sequence's first tile: the input, or the initial conv state."""
    in_seq = ii >= K
    xv = tl.load(x_base + x_offs - K * stride_x_t, mask=mask & in_seq[:, None], other=0.0)
    w_idx = (WIDTH - 1) + ii - K
    sv = tl.load(
        s_base + s_offs_c[None, :] + w_idx[:, None] * stride_s_k,
        mask=mask & (~in_seq)[:, None] & has_init,
        other=0.0,
    )
    return tl.where(in_seq[:, None], xv, sv).to(tl.float32)


@triton.jit
def _final_state_tap(
    x_ptr,
    s_base,
    s_offs_c,
    offs_c,
    c_mask,
    seq_start,
    seq_end,
    has_init,
    stride_x_t,
    stride_s_k,
    W: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """Final conv_states[slot, c, W] (input dtype), computed from the input."""
    seqlen = seq_end - seq_start
    full = seqlen >= WIDTH - 1
    from_x = full | (W >= (WIDTH - 1) - seqlen)
    src_t = tl.where(full, seq_end - (WIDTH - 1) + W, seq_start + W - ((WIDTH - 1) - seqlen))
    xv = tl.load(
        x_ptr + src_t.to(tl.int64) * stride_x_t + offs_c,
        mask=c_mask & from_x,
        other=0.0,
    )
    sv = tl.load(
        s_base + s_offs_c + (W + seqlen) * stride_s_k,
        mask=c_mask & (~from_x) & has_init,
        other=0.0,
    )
    return tl.where(from_x, xv, sv)
