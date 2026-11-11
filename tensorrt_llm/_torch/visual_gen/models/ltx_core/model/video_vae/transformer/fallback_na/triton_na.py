# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN001, ANN202, PLR0912, PLR0913, PLR0915
"""Triton 3D neighborhood attention (NATTEN ``na3d`` semantics).
Vendored from comfy-kitchen ``backends/triton/na.py`` (Apache-2.0) for DiffVAE hosts
without natten. One program handles a run of ``BLOCK_Q`` queries along W at a fixed
(t, h); online softmax, fp32 accumulation, no materialized scores.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Triton requires module-level JIT globals to be constexpr instances (not annotations).
_NEG_INF = tl.constexpr(-3.0e38)


@triton.jit
def _na3d_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    t_size,
    h_size,
    w_size,
    num_heads,
    bn_count,
    s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    scale,
    kt: tl.constexpr,
    kh: tl.constexpr,
    kw: tl.constexpr,
    causal_t: tl.constexpr,
    causal_h: tl.constexpr,
    causal_w: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    pid_w = tl.program_id(0)
    h_q = tl.program_id(1)
    pid_tbn = tl.program_id(2)

    # (frame, batch, head) share axis 2 and H owns axis 1, rather than folding T*H into axis 1:
    # gridDim.y/z are capped at 65535 and T*H passes that at production tile sizes (392*272 =
    # 106624), which fails the launch outright with "Triton Error [CUDA]: invalid argument".
    # Same layout as joint_triton, for the same reason.
    t_q = pid_tbn // bn_count
    pid_bn = pid_tbn % bn_count
    # int64 throughout: every individual stride fits in int32, so Triton specializes them there,
    # but t_q * s_t reaches ~1.3e10 on a 392-frame tile and would wrap -- an illegal memory access
    # rather than a wrong answer.
    base = (pid_bn // num_heads).to(tl.int64) * s_b + (pid_bn % num_heads).to(tl.int64) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = q_ptr + base + t_q.to(tl.int64) * s_t + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

    if causal_t:
        t_lo = tl.maximum(t_q - kt + 1, 0)
        t_hi = t_q + 1
    else:
        t_lo = tl.minimum(tl.maximum(t_q - kt // 2, 0), t_size - kt)
        t_hi = t_lo + kt
    if causal_h:
        h_lo = tl.maximum(h_q - kh + 1, 0)
        h_hi = h_q + 1
    else:
        h_lo = tl.minimum(tl.maximum(h_q - kh // 2, 0), h_size - kh)
        h_hi = h_lo + kh

    w_q = tl.where(w_valid, w_off, w_size - 1)
    if causal_w:
        w_start = tl.maximum(w_q - kw + 1, 0)
        w_end = w_q + 1
        blk_first = tl.minimum(pid_w * block_q, w_size - 1)
        blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
        w_lo = tl.maximum(blk_first - kw + 1, 0)
        w_hi = blk_last + 1
    else:
        w_start = tl.minimum(tl.maximum(w_q - kw // 2, 0), w_size - kw)
        w_end = w_start + kw
        blk_first = tl.minimum(pid_w * block_q, w_size - 1)
        blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
        w_lo = tl.minimum(tl.maximum(blk_first - kw // 2, 0), w_size - kw)
        w_hi = tl.minimum(tl.maximum(blk_last - kw // 2, 0), w_size - kw) + kw

    m_i = tl.full((block_q,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

    for tk in range(t_lo, t_hi):
        for hk in range(h_lo, h_hi):
            plane = base + tk.to(tl.int64) * s_t + hk.to(tl.int64) * s_h
            for wk0 in range(w_lo, w_hi, block_k):
                wk = wk0 + tl.arange(0, block_k)
                kmask = wk < w_hi
                kv_ptrs = plane + wk[:, None] * s_w + d_off[None, :]
                kv_mask = kmask[:, None] & d_mask[None, :]
                k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                if is_fp32:
                    s = tl.dot(q_blk, tl.trans(k_blk), input_precision="ieee") * scale
                else:
                    s = tl.dot(q_blk, tl.trans(k_blk)) * scale
                vis = (wk[None, :] >= w_start[:, None]) & (wk[None, :] < w_end[:, None]) & kmask[None, :]
                s = tl.where(vis, s, _NEG_INF)
                m_new = tl.maximum(m_i, tl.max(s, 1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1)
                v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                if is_fp32:
                    acc = acc * alpha[:, None] + tl.dot(p, v_blk, input_precision="ieee")
                else:
                    acc = acc * alpha[:, None] + tl.dot(p.to(v_blk.dtype), v_blk)
                m_i = m_new

    out = acc / tl.maximum(l_i, 1e-30)[:, None]
    out_ptrs = out_ptr + base + t_q.to(tl.int64) * s_t + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=w_valid[:, None] & d_mask[None, :])


def na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: list[int] | tuple[int, ...],
    is_causal: list[bool] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """3D neighborhood attention over ``(B, T, H, W, NH, HD)`` tensors."""
    batch, t, h, w, nh, hd = q.shape
    causal = [False, False, False] if is_causal is None else list(is_causal)
    kt, kh, kw = (k_ if c else min(k_, d) for k_, c, d in zip(kernel_size, causal, (t, h, w), strict=True))
    if scale is None:
        scale = hd**-0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)

    hd_p = max(16, triton.next_power_of_2(hd))
    block_q = 16
    block_k = max(16, min(32, triton.next_power_of_2(min(w, block_q + kw))))

    grid = (triton.cdiv(w, block_q), h, t * batch * nh)
    _na3d_kernel[grid](
        q,
        k,
        v,
        out,
        t,
        h,
        w,
        nh,
        batch * nh,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.stride(4),
        scale,
        kt=kt,
        kh=kh,
        kw=kw,
        causal_t=causal[0],
        causal_h=causal[1],
        causal_w=causal[2],
        hd=hd,
        hd_pad=hd_p,
        block_q=block_q,
        block_k=block_k,
        is_fp32=q.dtype == torch.float32,
        num_warps=4,
    )
    return out
