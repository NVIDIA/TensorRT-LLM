# ruff: noqa: ANN001, ANN202, PLR0913
"""Triton joint (video + keyframe) 3D neighborhood attention.
Same semantics as :mod:`joint_eager` -- see its module docstring for the window rule and
the two query kinds -- but flash-style: one program owns ``BLOCK_Q`` queries along W at a
fixed (frame-or-plane, h), keeps ``m/l/acc`` in registers, and never materializes scores.
Two kernels, because the two query kinds have different key topologies and different grids:
* ``_joint_video_kernel``  -- grid ``(cdiv(W, BQ), H, T * B * NH)``
* ``_joint_keyframe_kernel`` -- grid ``(cdiv(W, BQ), H, P * B * NH)``
The axis split matters at production sizes. Folding ``T * H`` into one grid axis passes
CUDA's 65535 cap on ``gridDim.y``/``z`` for a real tile (392 frames x 272 rows = 106624), so
the frame index shares axis 2 with batch and head while H owns axis 1 and the W block stays on
axis 0 -- keeping neighbouring programs on neighbouring addresses. Pointer arithmetic is
**int64**: every individual stride fits in int32, so Triton specializes them there, but
``t_q * s_t`` reaches ~6.5e9 on that same tile and would wrap.
Three things here differ from the vendored video-only ``triton_na`` kernel, and each is
load-bearing:
1. **Clamp-and-mask, not NATTEN's inward shift.** The T and H loops run a *fixed*
   ``kt``/``kh`` iterations (so nothing depends on runtime control flow, matching how
   upstream's Pallas kernel is written); out-of-range taps are clamped for the load and
   masked out of the softmax. There is no ``min(kernel, axis)`` shrink and no shift, so a
   boundary query sees fewer keys and the softmax renormalizes.
2. **The slot loop shares the same ``m_i``/``l_i``/``acc``.** Cross-stream keys are folded
   into the *same* online softmax as the local window -- one softmax per query over both
   sets, which is the whole point of "joint".
3. **The softmax weights are masked, not just the scores.** ``p = where(vis, exp(s - m), 0)``
   rather than relying on ``exp(-inf - -inf)``. A fully masked row is reachable here (an
   invalid keyframe plane attends to nothing at all), and with only the scores masked it
   would evaluate ``exp(0) = 1`` for every key and return the plain mean of V. The
   video-only kernel gets away without this because its rows are never wholly empty.
Keyframe count is a **runtime** argument, never ``constexpr``: plane counts vary per decode
and specializing on them would recompile per clip. Only ``num_slots`` (2) is ``constexpr``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.keyframes import (
    KEYFRAME_CONTEXT_SLOTS,
    keyframe_video_slots,
    video_keyframe_slots,
)

# Triton requires module-level JIT globals to be constexpr instances (not annotations).
_NEG_INF = tl.constexpr(-3.0e38)
_LSE_FLOOR = tl.constexpr(1e-30)


@triton.jit
def _fold_w_run(
    m_i,
    l_i,
    acc,
    q_blk,
    k_ptr,
    v_ptr,
    plane_base,
    row_ok,
    w_lo,
    w_hi,
    w_start,
    w_end,
    s_w,
    d_off,
    d_mask,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    """Fold one ``(frame|plane, h)`` row's W run into the running softmax state.
    ``row_ok`` is a scalar predicate for the whole row (the T/H tap or the slot being in
    range); ``w_start``/``w_end`` are per-query clamped window bounds. Shared by both
    kernels and by the local and slot loops, so every key set is folded by identical code.
    """
    for wk0 in range(w_lo, w_hi, block_k):
        wk = wk0 + tl.arange(0, block_k)
        kmask = wk < w_hi
        kv_ptrs = plane_base + wk[:, None] * s_w + d_off[None, :]
        # ``row_ok`` gates the *load*, not just the softmax: an out-of-range tap or an empty
        # slot has its index clamped to 0, and with no keyframe planes at all there is no
        # element 0 to read. Masking only ``vis`` would dereference it anyway.
        kv_mask = kmask[:, None] & d_mask[None, :] & row_ok
        k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
        k_t = tl.trans(k_blk)
        s = tl.dot(q_blk, k_t, input_precision="ieee") if is_fp32 else tl.dot(q_blk, k_t)
        vis = (wk[None, :] >= w_start[:, None]) & (wk[None, :] < w_end[:, None]) & kmask[None, :] & row_ok
        s = tl.where(vis, s, _NEG_INF)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        # Mask p, not only s: a wholly masked row must contribute 0, not exp(0) == 1.
        p = tl.where(vis, tl.exp(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
        if is_fp32:
            acc = acc * alpha[:, None] + tl.dot(p, v_blk, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v_blk.dtype), v_blk)
        m_i = m_new
    return m_i, l_i, acc


@triton.jit
def _joint_video_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kf_k_ptr,
    kf_v_ptr,
    slot_ptr,
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
    kf_s_b,
    kf_s_p,
    kt: tl.constexpr,
    kh: tl.constexpr,
    kw: tl.constexpr,
    num_slots: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    """Video queries: local ``Kt x Kh x Kw`` window plus ``num_slots`` keyframe planes."""
    pid_w = tl.program_id(0)
    h_q = tl.program_id(1)
    pid_tbn = tl.program_id(2)

    # (frame, batch, head) share axis 2 and H owns axis 1, rather than folding T*H into axis 1:
    # gridDim.y/z are capped at 65535 and T*H passes that at production tile sizes (392*272).
    # Axis 0 stays the W block so neighbouring programs keep touching neighbouring addresses.
    t_q = pid_tbn // bn_count
    pid_bn = pid_tbn % bn_count
    batch_i = pid_bn // num_heads
    head_i = pid_bn % num_heads
    # int64 throughout: s_t alone fits in int32, but t_q * s_t reaches ~6.5e9 on a 392-frame
    # tile at 1088x1920, and Triton would otherwise do that product in int32 and wrap.
    base = batch_i.to(tl.int64) * s_b + head_i.to(tl.int64) * s_n
    kf_base = batch_i.to(tl.int64) * kf_s_b + head_i.to(tl.int64) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = q_ptr + base + t_q.to(tl.int64) * s_t + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

    # Per-query clamped W window, and the union of them for this block of queries.
    w_q = tl.where(w_valid, w_off, 0)
    w_start = tl.maximum(w_q - kw // 2, 0)
    w_end = tl.minimum(w_q - kw // 2 + kw, w_size)
    blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
    w_lo = tl.maximum(pid_w * block_q - kw // 2, 0)
    w_hi = tl.minimum(blk_last - kw // 2 + kw, w_size)

    m_i = tl.full((block_q,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

    # Local video window: fixed kt*kh taps, clamped for the load and masked in the softmax.
    for d_t in range(kt):
        tk = t_q - kt // 2 + d_t
        t_ok = (tk >= 0) & (tk < t_size)
        tk_c = tl.maximum(tl.minimum(tk, t_size - 1), 0)
        for d_h in range(kh):
            hk = h_q - kh // 2 + d_h
            row_ok = t_ok & (hk >= 0) & (hk < h_size)
            hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
            m_i, l_i, acc = _fold_w_run(
                m_i,
                l_i,
                acc,
                q_blk,
                k_ptr,
                v_ptr,
                base + tk_c.to(tl.int64) * s_t + hk_c.to(tl.int64) * s_h,
                row_ok,
                w_lo,
                w_hi,
                w_start,
                w_end,
                s_w,
                d_off,
                d_mask,
                block_k=block_k,
                is_fp32=is_fp32,
            )

    # Nearest keyframe planes, into the same softmax state. No temporal offset: the plane
    # itself is the tap, and its visibility does not depend on kt.
    for slot in range(num_slots):
        plane = tl.load(slot_ptr + t_q * num_slots + slot)
        plane_ok = plane >= 0
        plane_c = tl.maximum(plane, 0)
        for d_h in range(kh):
            hk = h_q - kh // 2 + d_h
            row_ok = plane_ok & (hk >= 0) & (hk < h_size)
            hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
            m_i, l_i, acc = _fold_w_run(
                m_i,
                l_i,
                acc,
                q_blk,
                kf_k_ptr,
                kf_v_ptr,
                kf_base + plane_c.to(tl.int64) * kf_s_p + hk_c.to(tl.int64) * s_h,
                row_ok,
                w_lo,
                w_hi,
                w_start,
                w_end,
                s_w,
                d_off,
                d_mask,
                block_k=block_k,
                is_fp32=is_fp32,
            )

    out = acc / tl.maximum(l_i, _LSE_FLOOR)[:, None]
    out_ptrs = out_ptr + base + t_q.to(tl.int64) * s_t + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=w_valid[:, None] & d_mask[None, :])


@triton.jit
def _joint_keyframe_kernel(
    kf_q_ptr,
    kf_k_ptr,
    kf_v_ptr,
    k_ptr,
    v_ptr,
    slot_ptr,
    valid_ptr,
    out_ptr,
    h_size,
    w_size,
    num_heads,
    bn_count,
    s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    kf_s_b,
    kf_s_p,
    kh: tl.constexpr,
    kw: tl.constexpr,
    num_slots: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    """Keyframe queries: own plane only (no plane-to-plane) plus ``num_slots`` video frames.
    An invalid plane has no visible key at all -- its own-plane predicate is false and
    ``keyframe_video_slots`` gives it ``-1`` slots -- so ``l_i`` stays 0 and the store is
    exactly zero. That is the same observable result as upstream's overwrite-scores-with-0
    NaN guard.
    """
    pid_w = tl.program_id(0)
    h_q = tl.program_id(1)
    pid_pbn = tl.program_id(2)

    # Same axis split and the same int64 discipline as the video kernel.
    p_q = pid_pbn // bn_count
    pid_bn = pid_pbn % bn_count
    batch_i = pid_bn // num_heads
    head_i = pid_bn % num_heads
    base = batch_i.to(tl.int64) * s_b + head_i.to(tl.int64) * s_n
    kf_base = batch_i.to(tl.int64) * kf_s_b + head_i.to(tl.int64) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = (
        kf_q_ptr + kf_base + p_q.to(tl.int64) * kf_s_p + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    )
    q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

    w_q = tl.where(w_valid, w_off, 0)
    w_start = tl.maximum(w_q - kw // 2, 0)
    w_end = tl.minimum(w_q - kw // 2 + kw, w_size)
    blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
    w_lo = tl.maximum(pid_w * block_q - kw // 2, 0)
    w_hi = tl.minimum(blk_last - kw // 2 + kw, w_size)

    m_i = tl.full((block_q,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

    own_ok = tl.load(valid_ptr + p_q) != 0
    for d_h in range(kh):
        hk = h_q - kh // 2 + d_h
        row_ok = own_ok & (hk >= 0) & (hk < h_size)
        hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
        m_i, l_i, acc = _fold_w_run(
            m_i,
            l_i,
            acc,
            q_blk,
            kf_k_ptr,
            kf_v_ptr,
            kf_base + p_q.to(tl.int64) * kf_s_p + hk_c.to(tl.int64) * s_h,
            row_ok,
            w_lo,
            w_hi,
            w_start,
            w_end,
            s_w,
            d_off,
            d_mask,
            block_k=block_k,
            is_fp32=is_fp32,
        )

    for slot in range(num_slots):
        frame = tl.load(slot_ptr + p_q * num_slots + slot)
        frame_ok = frame >= 0
        frame_c = tl.maximum(frame, 0)
        for d_h in range(kh):
            hk = h_q - kh // 2 + d_h
            row_ok = frame_ok & (hk >= 0) & (hk < h_size)
            hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
            m_i, l_i, acc = _fold_w_run(
                m_i,
                l_i,
                acc,
                q_blk,
                k_ptr,
                v_ptr,
                base + frame_c.to(tl.int64) * s_t + hk_c.to(tl.int64) * s_h,
                row_ok,
                w_lo,
                w_hi,
                w_start,
                w_end,
                s_w,
                d_off,
                d_mask,
                block_k=block_k,
                is_fp32=is_fp32,
            )

    out = acc / tl.maximum(l_i, _LSE_FLOOR)[:, None]
    out_ptrs = (
        out_ptr + kf_base + p_q.to(tl.int64) * kf_s_p + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    )
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=w_valid[:, None] & d_mask[None, :])


def _check_shared_layout(video: torch.Tensor, keyframes: torch.Tensor) -> None:
    """Both streams must be contiguous and agree on H/W/NH/HD.
    The kernels pass one set of ``s_h``/``s_w``/``s_n`` strides for both streams, which is
    only sound because of this -- so it is checked rather than assumed.
    """
    if video.shape[2:] != keyframes.shape[2:]:
        raise ValueError(
            f"video and keyframe streams must agree on H/W/NH/HD, got {tuple(video.shape)} vs {tuple(keyframes.shape)}"
        )
    if not video.is_contiguous() or not keyframes.is_contiguous():
        raise ValueError("joint Triton NA needs both streams contiguous")
    if video.stride()[2:5] != keyframes.stride()[2:5]:
        raise ValueError(
            f"H/W/NH strides must match across streams, got {video.stride()[2:5]} vs {keyframes.stride()[2:5]}"
        )


def joint_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: tuple[int, int, int],
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint neighborhood attention over a video volume and a keyframe plane stack.
    Signature-compatible with :func:`joint_eager.joint_na3d` so the two are drop-in
    swappable and can be diffed directly.
    Args:
        q, k, v: ``(B, T, H, W, NH, HD)`` video stream, already normed/scaled/RoPE'd.
        keyframe_q, keyframe_k, keyframe_v: ``(B, P, H, W, NH, HD)`` keyframe stream.
        keyframe_times: ``(P,)`` float32 plane times, same origin as the video RoPE.
        keyframe_valid: ``(P,)`` bool.
        kernel_size: ``(Kt, Kh, Kw)``.
        num_slots: cross-stream slots per query.
    Returns:
        ``(video_out, keyframe_out)``, each shaped like its stream's ``q``.
    """
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    keyframe_q = keyframe_q.contiguous()
    keyframe_k = keyframe_k.contiguous()
    keyframe_v = keyframe_v.contiguous()
    _check_shared_layout(q, keyframe_q)

    batch, time, height, width, heads, head_dim = q.shape
    planes = keyframe_q.shape[1]
    if keyframe_times.shape != (planes,) or keyframe_valid.shape != (planes,):
        raise ValueError(
            f"keyframe_times/keyframe_valid must be ({planes},), got "
            f"{tuple(keyframe_times.shape)} / {tuple(keyframe_valid.shape)}"
        )
    kernel_t, kernel_h, kernel_w = kernel_size

    # Slot tables are built here rather than passed in: every backend must agree on
    # selection, so they come from the one shared implementation.
    video_slots = video_keyframe_slots(keyframe_times, keyframe_valid, time, num_slots)
    keyframe_slots = keyframe_video_slots(keyframe_times, keyframe_valid, time, num_slots)
    video_slots = video_slots.to(device=q.device, dtype=torch.int32).contiguous()
    keyframe_slots = keyframe_slots.to(device=q.device, dtype=torch.int32).contiguous()
    valid_i32 = keyframe_valid.to(device=q.device, dtype=torch.int32).contiguous()

    video_out = torch.empty_like(q)
    keyframe_out = torch.empty_like(keyframe_q)

    hd_pad = max(16, triton.next_power_of_2(head_dim))
    block_q = 16
    block_k = max(16, min(32, triton.next_power_of_2(min(width, block_q + kernel_w))))
    is_fp32 = q.dtype == torch.float32
    strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4))
    kf_strides = (keyframe_q.stride(0), keyframe_q.stride(1))

    bn_count = batch * heads
    _joint_video_kernel[(triton.cdiv(width, block_q), height, time * bn_count)](
        q,
        k,
        v,
        keyframe_k,
        keyframe_v,
        video_slots,
        video_out,
        time,
        height,
        width,
        heads,
        bn_count,
        *strides,
        *kf_strides,
        kt=kernel_t,
        kh=kernel_h,
        kw=kernel_w,
        num_slots=num_slots,
        hd=head_dim,
        hd_pad=hd_pad,
        block_q=block_q,
        block_k=block_k,
        is_fp32=is_fp32,
        num_warps=4,
    )
    _joint_keyframe_kernel[(triton.cdiv(width, block_q), height, planes * bn_count)](
        keyframe_q,
        keyframe_k,
        keyframe_v,
        k,
        v,
        keyframe_slots,
        valid_i32,
        keyframe_out,
        height,
        width,
        heads,
        bn_count,
        *strides,
        *kf_strides,
        kh=kernel_h,
        kw=kernel_w,
        num_slots=num_slots,
        hd=head_dim,
        hd_pad=hd_pad,
        block_q=block_q,
        block_k=block_k,
        is_fp32=is_fp32,
        num_warps=4,
    )
    return video_out, keyframe_out
