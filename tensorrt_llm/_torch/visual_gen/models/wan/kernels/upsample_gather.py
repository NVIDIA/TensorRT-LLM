# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The upsample phase gather in CuTe DSL: the nearest-2x upsample + 3x3 conv runs as a 2x2 conv
on the low-resolution grid with ``4 * Cout`` output channels (one block per output phase, see
``wino_vae_kernels.pack_upsample_weights``); this kernel scatters the phases into the 2x
resolution frames, adds the conv bias when the conv ran without it, and adds the DupUp3D
shortcut while writing.

One thread owns one source pixel ``(b, tr, i, j)`` and 8 output channels, and produces the
``factor_t x 2 x 2`` output pixels that descend from it: ``factor_t * 4`` 16-byte loads from the
conv output, one shortcut span load, ``factor_t * 4`` 16-byte stores in channels-last order.
Consecutive threads hold consecutive channel vectors of one pixel, so every warp access is
contiguous. Rounding: ``round(temp + bias)``, then ``round(that + shortcut)``, per element.
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
    bf16_hi,
    bf16_lo,
    compile_cached,
    pack2,
    store_words,
    u32_view,
    vec_view,
)

THREADS = 256


class UpsampleGatherKernel:
    def __init__(
        self, fp16: bool, factor_t: int, repeats: int, shortcut: bool, has_bias: bool
    ) -> None:
        self.fp16 = fp16
        self.dtype = Float16 if fp16 else BFloat16
        self.factor_t = factor_t
        self.factor = factor_t * 4
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
        ).launch(
            grid=[(items + THREADS - 1) // THREADS, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

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
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        idx = Int32(bid) * THREADS + Int32(tid)
        if idx < items:
            gt = vec_view(temp_ptr)
            gb = vec_view(bias_ptr)
            uo = u32_view(out_ptr)
            gs = cute.make_tensor(
                sc_ptr, cute.make_layout((self.chunk, 1 << 40), stride=(1, self.chunk))
            )
            ft = self.factor_t
            cvec = cout // CHANNELS_PER_VECTOR
            ov = idx % cvec
            o = ov * CHANNELS_PER_VECTOR
            pix = idx // cvec
            j = pix % width
            pix = pix // width
            i = pix % height
            pix = pix // height
            frames_r = (frames + crop) // ft
            tr = pix % frames_r
            b = pix // frames_r

            bias = None
            if cutlass.const_expr(self.has_bias):
                # Scalars extracted here, once: an element read from the vector inside the frame
                # branches would be materialized in the first branch only.
                loaded_bias = gb[(None, ov)].load().to(Float32)
                bias = [loaded_bias[k] for k in range(CHANNELS_PER_VECTOR)]
            span = []
            if cutlass.const_expr(self.shortcut):
                pix_base = ((Int64(b) * frames_r + tr) * height + i) * width + j
                first = (pix_base * cin + (Int64(o) * self.factor) // self.repeats) // self.chunk
                for c in cutlass.range_constexpr(self.span // self.chunk):
                    loaded = gs[(None, first + c)].load().to(Float32)
                    for e in cutlass.range_constexpr(self.chunk):
                        span.append(loaded[e])

            temp_pix = Int64(height + 1) * (width + 1)
            temp_row = Int64(width + 1) * (4 * cvec)  # in 16-byte vectors
            out_row = Int64(2 * width) * cvec
            out_frame = Int64(2 * height) * out_row
            # Every phase load of every frame is issued before the first store (the stores could
            # alias the loads as far as the compiler knows). On the cropped first chunk the
            # dropped frame reads frame 0 and skips its stores.
            frame = []
            phases = []
            for a in cutlass.range_constexpr(ft):
                f = tr * ft + a - crop  # output frame; negative on the cropped first chunk
                bt = Int64(b) * frames + cutlass.max(f, 0)
                temp_base = (bt * temp_pix + Int64(i) * (width + 1) + j) * (4 * cvec) + ov
                for ph in cutlass.range_constexpr(4):
                    pp, qq = ph // 2, ph % 2
                    t_idx = temp_base + pp * temp_row + qq * (4 * cvec) + ph * cvec
                    phases.append(gt[(None, t_idx)].load().to(Float32))
                frame.append((f, bt))
            for a in cutlass.range_constexpr(ft):
                f, bt = frame[a]
                out_base = bt * out_frame + Int64(2 * i) * out_row + Int64(2 * j) * cvec + ov
                words_ph = []
                for ph in cutlass.range_constexpr(4):
                    pp, qq = ph // 2, ph % 2
                    v = phases[a * 4 + ph]
                    vals = [v[k] for k in range(CHANNELS_PER_VECTOR)]
                    if cutlass.const_expr(self.has_bias):
                        vals = self._round8([vals[k] + bias[k] for k in range(CHANNELS_PER_VECTOR)])
                    if cutlass.const_expr(self.shortcut):
                        s = (a * 2 + pp) * 2 + qq
                        vals = [
                            vals[k] + span[(k * self.factor + s) // self.repeats]
                            for k in range(CHANNELS_PER_VECTOR)
                        ]
                    words_ph.append(
                        [
                            pack2(vals[2 * w], vals[2 * w + 1], self.fp16)
                            for w in range(CHANNELS_PER_VECTOR // 2)
                        ]
                    )
                if f >= 0:
                    for ph in cutlass.range_constexpr(4):
                        pp, qq = ph // 2, ph % 2
                        store_words(uo, out_base + pp * out_row + qq * cvec, words_ph[ph])

    def _round8(self, vals: list) -> list:
        """Eight floats rounded to the activation dtype and back: packed pairs for bf16 (the
        per-element bf16 conversion takes the slow ``F2F`` path), the vector conversion for fp16
        (already packed)."""
        if not self.fp16:
            out = []
            for w in range(CHANNELS_PER_VECTOR // 2):
                word = pack2(vals[2 * w], vals[2 * w + 1])
                out += [bf16_lo(word), bf16_hi(word)]
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
    dtype = Float16 if fp16 else BFloat16

    def compile_fn():
        ptrs = [make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16) for _ in range(4)]
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        return cute.compile(
            kernel.wrapper, *ptrs, *([Int32(1)] * 7), stream, options="--enable-tvm-ffi"
        )

    name = f"upsample_gather[{int(fp16)},{factor_t},{repeats},{int(shortcut)},{int(bias)}]"
    return compile_cached("trtllm_wan_vae_upsample", name, compile_fn, [__file__])


def can_upsample_gather(cout: int, factor_t: int, repeats: int) -> bool:
    """Shapes the gather covers: ``Cout`` a multiple of 8, a temporal factor of 1 or 2 and a
    power-of-two channel repeat up to 32."""
    return (
        cout % 8 == 0 and factor_t in (1, 2) and 1 <= repeats <= 32 and not repeats & (repeats - 1)
    )


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
    conv ``bias`` (``Cout`` values) and the DupUp3D ``shortcut`` (``[B, Cin, Tr, H, W]``
    channels_last_3d, ``Tr * factor_t == T + crop``, ``Cin * repeats == 4 * Cout * factor_t``)
    added while writing; ``crop`` leading output frames of the shortcut's expansion are
    dropped (the first chunk)."""
    if temp.dim() != 4 or temp.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("temp must be a bf16 or fp16 [B*T, 4*Cout, H+1, W+1] tensor")
    if not temp.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("temp must be channels-last")
    if temp.shape[0] != batch * frames or temp.shape[1] % 4:
        raise ValueError("temp must hold B*T frames of 4*Cout channels")
    cout = temp.shape[1] // 4
    height, width = temp.shape[2] - 1, temp.shape[3] - 1
    if shortcut is None:
        # Without a shortcut every output frame is its own source frame.
        factor_t, repeats, crop = 1, 1, 0
    if height < 1 or width < 1 or not can_upsample_gather(cout, factor_t, repeats):
        raise ValueError("unsupported upsample gather shape")
    cin = 0
    if shortcut is not None:
        s = shortcut
        if s.dim() != 5 or s.dtype != temp.dtype:
            raise ValueError("shortcut must be a 5D tensor of temp's dtype")
        if not s.is_contiguous(memory_format=torch.channels_last_3d):
            raise ValueError("shortcut must be channels_last_3d")
        if s.shape[0] != batch or tuple(s.shape[3:]) != (height, width):
            raise ValueError("shortcut spatial mismatch")
        if s.shape[2] * factor_t != frames + crop or s.shape[1] * repeats != cout * factor_t * 4:
            raise ValueError("shortcut frames or channels do not match")
        cin = s.shape[1]
    if bias is not None:
        bias = bias.contiguous()
        if bias.dtype != temp.dtype or bias.numel() != cout:
            raise ValueError("bias must be [Cout] values of temp's dtype")
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
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
    return out
