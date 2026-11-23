# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The upsample phase gather in CuTe DSL: the nearest-2x upsample + 3x3 conv runs as a 2x2 conv
on the low-resolution grid with ``4 * Cout`` output channels (one block per output phase, see
``wino_vae_kernels.pack_upsample_weights``); this kernel scatters the phases into the 2x
resolution frames, adds the conv bias when the conv ran without it, and adds the DupUp3D
shortcut while writing.

One thread owns one source pixel ``(b, tr, i, j)`` and 8 output channels, and produces the
``factor_t x 2 x 2`` output pixels that descend from it: ``factor_t * 4`` 16-byte loads from the
conv output, the thread's shortcut span in loads of up to 16 bytes (at most 8), and
``factor_t * 4`` 16-byte stores in channels-last order. Consecutive threads hold consecutive
channel vectors of one pixel, so every warp access is contiguous. Rounding: ``round(temp +
bias)``, then ``round(that + shortcut)``, per element.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64

from .common import (
    CHANNELS_PER_VECTOR,
    THREADS,
    check_aligned,
    check_bias,
    compile_cached,
    current_stream,
    flat_view,
    grid_1d,
    linear_thread_index,
    pack_words,
    round_pair_bf16,
    store_words,
    u32_view,
    vec_view,
)

PHASES = 4  # output pixels (2 x 2) per source pixel and frame


class UpsampleGatherKernel:
    def __init__(
        self, fp16: bool, factor_t: int, repeats: int, shortcut: bool, has_bias: bool
    ) -> None:
        self.fp16 = fp16
        self.dtype = Float16 if fp16 else BFloat16
        self.factor_t = factor_t
        self.factor = factor_t * PHASES
        self.repeats = repeats
        self.shortcut = shortcut
        self.has_bias = has_bias
        # Shortcut channels of one thread: a contiguous span of 8 * factor / repeats values
        # (at least one), loaded in chunks of up to one 16-byte vector.
        self.span = max(1, CHANNELS_PER_VECTOR * self.factor // repeats)
        self.chunk = min(self.span, CHANNELS_PER_VECTOR)

    @cute.jit
    def wrapper(
        self,
        temp_ptr: cute.Pointer,
        sc_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        frames: Int32,
        height: Int32,
        width: Int32,
        cout: Int32,
        cin: Int32,
        crop: Int32,
        items: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            temp_ptr, sc_ptr, bias_ptr, out_ptr, frames, height, width, cout, cin, crop, items
        ).launch(grid=grid_1d(items), block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        temp_ptr: cute.Pointer,
        sc_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        frames: Int32,
        height: Int32,
        width: Int32,
        cout: Int32,
        cin: Int32,
        crop: Int32,
        items: Int32,
    ):
        idx = linear_thread_index()
        if idx < items:
            temp_vecs = vec_view(temp_ptr)
            bias_vecs = vec_view(bias_ptr)
            out_words = u32_view(out_ptr)
            shortcut_chunks = flat_view(sc_ptr, self.chunk)
            cvec = cout // CHANNELS_PER_VECTOR
            out_vec, channel0, b, tr, i, j, frames_r = self._decode_source_pixel(
                idx, cvec, frames, height, width, crop
            )
            bias = self._load_bias(bias_vecs, out_vec)
            span = self._load_shortcut_span(
                shortcut_chunks, b, tr, i, j, frames_r, height, width, cin, channel0
            )

            temp_pix = Int64(height + 1) * (width + 1)
            temp_row = Int64(width + 1) * (PHASES * cvec)  # in 16-byte vectors
            out_row = Int64(2 * width) * cvec
            out_frame = Int64(2 * height) * out_row
            # Every phase load of every frame is issued before the first store (the stores could
            # alias the loads as far as the compiler knows). On the cropped first chunk the
            # dropped frame reads frame 0 and skips its stores.
            frame, phases = self._load_phases(
                temp_vecs, b, tr, i, j, out_vec, cvec, frames, width, crop, temp_pix, temp_row
            )
            for a in cutlass.range_constexpr(self.factor_t):
                f, bt = frame[a]
                out_base = bt * out_frame + Int64(2 * i) * out_row + Int64(2 * j) * cvec + out_vec
                words = [
                    self._phase_words(phases[a * PHASES + ph], bias, span, a, ph)
                    for ph in range(PHASES)
                ]
                if f >= 0:
                    for ph in cutlass.range_constexpr(PHASES):
                        phase_row, phase_col = divmod(ph, 2)
                        store_words(
                            out_words, out_base + phase_row * out_row + phase_col * cvec, words[ph]
                        )

    def _decode_source_pixel(self, idx, cvec, frames, height, width, crop):
        """Thread index -> (channel vector, its first channel, batch, source frame, row, column,
        source frames per batch)."""
        out_vec = idx % cvec
        channel0 = out_vec * CHANNELS_PER_VECTOR
        pix = idx // cvec
        j = pix % width
        rows = pix // width
        i = rows % height
        frames_all = rows // height
        frames_r = (frames + crop) // self.factor_t
        tr = frames_all % frames_r
        b = frames_all // frames_r
        return out_vec, channel0, b, tr, i, j, frames_r

    def _load_bias(self, bias_vecs, out_vec) -> list | None:
        """The thread's 8 bias values as floats, or ``None`` without a bias."""
        if not self.has_bias:
            return None
        # Scalars extracted here, once: an element read from the vector inside the frame
        # branches would be materialized in the first branch only.
        loaded = bias_vecs[(None, out_vec)].load().to(Float32)
        return [loaded[k] for k in range(CHANNELS_PER_VECTOR)]

    def _load_shortcut_span(
        self, shortcut_chunks, b, tr, i, j, frames_r, height, width, cin, channel0
    ) -> list:
        """The shortcut values the thread's output channels repeat, as floats (empty without a
        shortcut)."""
        span = []
        if not self.shortcut:
            return span
        pix_base = ((Int64(b) * frames_r + tr) * height + i) * width + j
        first = (pix_base * cin + (Int64(channel0) * self.factor) // self.repeats) // self.chunk
        for c in range(self.span // self.chunk):
            loaded = shortcut_chunks[(None, first + c)].load().to(Float32)
            for e in range(self.chunk):
                span.append(loaded[e])
        return span

    def _load_phases(
        self, temp_vecs, b, tr, i, j, out_vec, cvec, frames, width, crop, temp_pix, temp_row
    ):
        """The four phase vectors of every output frame of the thread, as fp32 vectors in
        (frame, phase) order, and per frame ``(output frame, its batch-frame index)``; the
        output frame is negative for the frame dropped on the cropped first chunk."""
        frame = []
        phases = []
        for a in range(self.factor_t):
            f = tr * self.factor_t + a - crop
            bt = Int64(b) * frames + cutlass.max(f, 0)
            temp_base = (bt * temp_pix + Int64(i) * (width + 1) + j) * (PHASES * cvec) + out_vec
            for ph in range(PHASES):
                phase_row, phase_col = divmod(ph, 2)
                t_idx = temp_base + phase_row * temp_row + phase_col * (PHASES * cvec) + ph * cvec
                phases.append(temp_vecs[(None, t_idx)].load().to(Float32))
            frame.append((f, bt))
        return frame, phases

    def _phase_words(self, v, bias: list | None, span: list, a: int, ph: int) -> list:
        """Output pixel ``ph`` of output frame ``a``: the phase vector ``v`` plus the bias
        (rounded) plus the shortcut, as packed words."""
        vals = [v[k] for k in range(CHANNELS_PER_VECTOR)]
        if self.has_bias:
            vals = self._round8([vals[k] + bias[k] for k in range(CHANNELS_PER_VECTOR)])
        if self.shortcut:
            phase_row, phase_col = divmod(ph, 2)
            s = (a * 2 + phase_row) * 2 + phase_col
            vals = [
                vals[k] + span[(k * self.factor + s) // self.repeats]
                for k in range(CHANNELS_PER_VECTOR)
            ]
        return pack_words(vals, self.fp16)

    def _round8(self, vals: list) -> list:
        """Eight floats rounded to the activation dtype and back: packed pairs for bf16 (the
        per-element bf16 conversion takes the slow ``F2F`` path), the vector conversion for fp16
        (already packed)."""
        if not self.fp16:
            out = []
            for w in range(CHANNELS_PER_VECTOR // 2):
                _, lo, hi = round_pair_bf16(vals[2 * w], vals[2 * w + 1])
                out += [lo, hi]
            return out
        t = cute.make_rmem_tensor((CHANNELS_PER_VECTOR,), Float32)
        for k in range(CHANNELS_PER_VECTOR):
            t[k] = vals[k]
        r = t.load().to(Float16).to(Float32)
        return [r[k] for k in range(CHANNELS_PER_VECTOR)]


@functools.lru_cache(maxsize=None)
def _gather_kernel(
    device: int, fp16: bool, factor_t: int, repeats: int, shortcut: bool, bias: bool
):
    from cutlass.cute.runtime import make_ptr

    kernel = UpsampleGatherKernel(fp16, factor_t, repeats, shortcut, bias)

    def compile_fn():
        ptrs = [
            make_ptr(kernel.dtype, 16, cute.AddressSpace.gmem, assumed_align=16) for _ in range(4)
        ]
        return cute.compile(
            kernel.wrapper,
            *ptrs,
            *([Int32(1)] * 7),
            current_stream(device),
            options="--enable-tvm-ffi",
        )

    name = f"upsample_gather[{int(fp16)},{factor_t},{repeats},{int(shortcut)},{int(bias)}]"
    return compile_cached("trtllm_wan_vae_upsample", name, compile_fn, [__file__])


def can_upsample_gather(cout: int, factor_t: int, repeats: int) -> bool:
    """Shapes the gather covers: ``Cout`` a multiple of 8, a temporal factor of 1 or 2 and a
    power-of-two channel repeat up to 32."""
    return (
        cout % CHANNELS_PER_VECTOR == 0
        and factor_t in (1, 2)
        and 1 <= repeats <= 32
        and not repeats & (repeats - 1)
    )


def _check_temp(temp: torch.Tensor, batch: int, frames: int) -> None:
    """The packed conv output is a channels-last bf16 / fp16 ``[B*T, 4*Cout, H+1, W+1]``."""
    if temp.dim() != 4 or temp.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("temp must be a bf16 or fp16 [B*T, 4*Cout, H+1, W+1] tensor")
    if not temp.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("temp must be channels-last")
    if temp.shape[0] != batch * frames or temp.shape[1] % 4:
        raise ValueError("temp must hold B*T frames of 4*Cout channels")


def _check_shortcut(
    s: torch.Tensor,
    dtype: torch.dtype,
    batch: int,
    height: int,
    width: int,
    expanded: tuple[int, int],
    want: tuple[int, int],
) -> None:
    """The DupUp3D shortcut is a channels-last-3d ``[B, Cin, Tr, H, W]`` of ``dtype`` whose
    expansion ``(Tr * factor_t, Cin * repeats)`` = ``expanded`` matches ``want`` = ``(T +
    crop, 4 * Cout * factor_t)``."""
    if s.dim() != 5 or s.dtype != dtype:
        raise ValueError("shortcut must be a 5D tensor of temp's dtype")
    if not s.is_contiguous(memory_format=torch.channels_last_3d):
        raise ValueError("shortcut must be channels_last_3d")
    if s.shape[0] != batch or tuple(s.shape[3:]) != (height, width):
        raise ValueError("shortcut spatial mismatch")
    if expanded != want:
        raise ValueError("shortcut frames or channels do not match")


def upsample_gather(
    temp: torch.Tensor,
    shortcut: torch.Tensor | None,
    batch: int,
    frames: int,
    factor_t: int = 1,
    repeats: int = 1,
    crop: int = 0,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Phase gather of the packed 2x2 conv output ``temp [B*T, 4*Cout, H+1, W+1]``
    (channels-last, bf16 or fp16) into ``[B, Cout, T, 2H, 2W]`` (channels_last_3d), with the
    conv ``bias`` (contiguous ``[Cout]`` of temp's dtype) and the DupUp3D ``shortcut`` (``[B,
    Cin, Tr, H, W]`` channels_last_3d, ``Tr * factor_t == T + crop``, ``Cin * repeats == 4 *
    Cout * factor_t``) added while writing; ``crop`` leading output frames of the shortcut's
    expansion are dropped (the first chunk). All inputs must be 16-byte aligned."""
    _check_temp(temp, batch, frames)
    cout = temp.shape[1] // 4
    height, width = temp.shape[2] - 1, temp.shape[3] - 1
    if shortcut is None:
        # Without a shortcut every output frame is its own source frame.
        factor_t, repeats, crop = 1, 1, 0
    if height < 1 or width < 1 or not can_upsample_gather(cout, factor_t, repeats):
        raise ValueError("unsupported upsample gather shape")
    if shortcut is not None:
        expanded = (shortcut.shape[2] * factor_t, shortcut.shape[1] * repeats)
        want = (frames + crop, cout * factor_t * 4)
        _check_shortcut(shortcut, temp.dtype, batch, height, width, expanded, want)
    cin = shortcut.shape[1] if shortcut is not None else 0
    check_bias(bias, cout, temp.dtype, "Cout")
    check_aligned(16, temp=temp, shortcut=shortcut, bias=bias)
    out = torch.empty(
        (batch, cout, frames, 2 * height, 2 * width),
        dtype=temp.dtype,
        device=temp.device,
        memory_format=torch.channels_last_3d,
    )
    frames_r = (frames + crop) // factor_t
    items = batch * frames_r * height * width * (cout // CHANNELS_PER_VECTOR)
    if items == 0:
        return out
    if items >= 2**31:
        raise ValueError("upsample_gather: too many source pixels")
    device = temp.device.index
    fn = _gather_kernel(
        device,
        temp.dtype == torch.float16,
        factor_t,
        repeats,
        shortcut is not None,
        bias is not None,
    )
    fn(
        temp.data_ptr(),
        shortcut.data_ptr() if shortcut is not None else 0,
        bias.data_ptr() if bias is not None else 0,
        out.data_ptr(),
        frames,
        height,
        width,
        cout,
        cin,
        crop,
        items,
        current_stream(device),
    )
    return out
