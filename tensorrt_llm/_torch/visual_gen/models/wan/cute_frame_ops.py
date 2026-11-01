# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WinoVae frame operations in CuTe DSL (bf16, batch-1 channels-last frames ``[T, P, C]``).

``avg_down_add``: ``y = x (+ bias) + AvgDown3D(r)``, the residual down block's shortcut add.
AvgDown3D groups the ``cin * factor`` values of an output pixel's window (input channel
slowest, then the temporal tap, then the two spatial taps; ``factor = factor_t * factor_s^2``)
into ``cout`` groups of ``group_size`` consecutive values and takes each group's mean. When a
group lies inside one input channel's window (``factor % group_size == 0``, every Wan
configuration), a thread owns one output pixel and 8 input channels: it reads each window tap
as one 16-byte vector (the residual is read exactly once) and produces the ``8 * factor /
group_size`` output channels those inputs feed.

Numerics, per output channel: the group's values summed in fp32 in window order (temporal pad
taps add nothing), the mean ``sum * (1 / group_size)`` rounded to bf16, then ``x + bias`` in
fp32 plus the mean, rounded once.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32

from .cute_common import (
    VEC,
    compile_cached,
    load_words,
    pack2,
    store_words,
    u32_view,
    unpack_bf16,
    vec_view,
)

THREADS = 256


class AvgDownAddKernel:
    def __init__(
        self, cin: int, cout: int, factor_t: int, factor_s: int, group_size: int, has_bias: bool
    ) -> None:
        self.cin, self.cout = cin, cout
        self.factor_t, self.factor_s = factor_t, factor_s
        self.factor = factor_t * factor_s * factor_s
        self.group_size = group_size
        self.per_input = self.factor // group_size  # output channels per input channel
        self.has_bias = has_bias
        self.cvec = cin // VEC

    @cute.jit
    def wrapper(
        self,
        x_ptr: cute.Pointer,
        r_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        out_positions: Int32,
        out_width: Int32,
        in_width: Int32,
        in_rows: Int32,
        pad_t: Int32,
        items: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x_ptr, r_ptr, bias_ptr, y_ptr, out_positions, out_width, in_width, in_rows, pad_t, items
        ).launch(
            grid=[(items + THREADS - 1) // THREADS, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x_ptr: cute.Pointer,
        r_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        out_positions: Int32,
        out_width: Int32,
        in_width: Int32,
        in_rows: Int32,
        pad_t: Int32,
        items: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        idx = Int32(bid) * THREADS + Int32(tid)
        if idx < items:
            ux = u32_view(x_ptr)
            ub = u32_view(bias_ptr)
            uy = u32_view(y_ptr)
            gr = vec_view(r_ptr)
            fs, ft = self.factor_s, self.factor_t
            cv = idx % self.cvec
            pix = idx // self.cvec
            t_out = pix // out_positions
            sp = pix - t_out * out_positions
            i_out = sp // out_width
            j_out = sp - i_out * out_width

            # The window taps in window order (temporal, then the two spatial), 8 channels each.
            taps = []
            for a in cutlass.range_constexpr(ft):
                t_in = t_out * ft + a - pad_t
                for b in cutlass.range_constexpr(fs):
                    for d in cutlass.range_constexpr(fs):
                        row = (t_in * in_rows + i_out * fs + b) * in_width + j_out * fs + d
                        buf = cute.make_rmem_tensor((VEC,), BFloat16)
                        if t_in >= 0:
                            buf.store(gr[(None, cutlass.Int64(row) * self.cvec + cv)].load())
                        else:
                            buf.store(cute.full((VEC,), 0.0, Float32).to(BFloat16))
                        taps.append(buf.load().to(Float32))

            inv = Float32(1.0 / self.group_size)
            for w in cutlass.range_constexpr(self.per_input):
                ov = cutlass.Int64(pix) * (self.cout // VEC) + cv * self.per_input + w
                xs = unpack_bf16(load_words(ux, ov))
                bs = None
                if cutlass.const_expr(self.has_bias):
                    bs = unpack_bf16(load_words(ub, cv * self.per_input + w))
                means = []
                for k in cutlass.range_constexpr(VEC):
                    o = w * VEC + k  # output channel, counted from this thread's first one
                    e, q = o // self.per_input, o % self.per_input
                    acc = Float32(0.0)
                    for g in cutlass.range_constexpr(self.group_size):
                        acc = acc + taps[q * self.group_size + g][e]
                    means.append(acc * inv)
                words = []
                for i in cutlass.range_constexpr(VEC // 2):
                    mw = pack2(means[2 * i], means[2 * i + 1])
                    rounded = unpack_bf16([mw])
                    s = []
                    for h in cutlass.range_constexpr(2):
                        xb = xs[2 * i + h]
                        if cutlass.const_expr(self.has_bias):
                            xb = xb + bs[2 * i + h]
                        s.append(xb + rounded[h])
                    words.append(pack2(s[0], s[1]))
                store_words(uy, ov, words)


@functools.lru_cache(maxsize=None)
def _avg_down_add_kernel(
    device: int, cin: int, cout: int, factor_t: int, factor_s: int, group_size: int, bias: bool
):
    from cutlass.cute.runtime import make_ptr

    kernel = AvgDownAddKernel(cin, cout, factor_t, factor_s, group_size, bias)

    def compile_fn():
        ptrs = [make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16) for _ in range(4)]
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        return cute.compile(
            kernel.wrapper, *ptrs, *([Int32(1)] * 6), stream, options="--enable-tvm-ffi"
        )

    name = f"avg_down_add[{cin},{cout},{factor_t},{factor_s},{group_size},{int(bias)}]"
    return compile_cached("trtllm_wan_vae_frame_ops", name, compile_fn, [__file__])


def avg_down_add(
    x: torch.Tensor,
    r: torch.Tensor,
    out_height: int,
    out_width: int,
    in_width: int,
    factor_t: int,
    factor_s: int,
    group_size: int,
    pad_t: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x (+ bias) + AvgDown3D(r)`` in frame layout: ``x [T', P', Cout]``, ``r [T, P, Cin]``
    contiguous bf16, ``pad_t`` zero frames in front of ``r``; ``bias`` (``Cout`` values) is the
    downsample conv's bias when that conv ran without it. Returns ``y`` like ``x``."""
    for name, t in (("x", x), ("r", r), ("bias", bias)):
        if t is not None and (t.dtype != torch.bfloat16 or not t.is_contiguous()):
            raise ValueError(f"{name} must be contiguous bf16")
    if x.dim() != 3 or r.dim() != 3:
        raise ValueError("x and r must be [T, P, C] frames")
    out_frames, out_positions, cout = x.shape
    in_frames, _, cin = r.shape
    factor = factor_t * factor_s * factor_s
    if out_positions != out_height * out_width:
        raise ValueError("x positions must equal out_height * out_width")
    if in_frames + pad_t != out_frames * factor_t:
        raise ValueError("frame counts do not match the temporal factor")
    if cin * factor != cout * group_size or factor % group_size or cin % VEC or cout % VEC:
        raise ValueError(
            "avg_down_add needs cin * factor == cout * group_size, group_size dividing factor "
            "and channel counts that are multiples of 8"
        )
    if bias is not None and bias.numel() != cout:
        raise ValueError("bias must have cout values")
    y = torch.empty_like(x)
    items = out_frames * out_positions * (cin // VEC)
    if items == 0:
        return y
    if items >= 2**31:
        raise ValueError("avg_down_add: too many output pixels for one launch")
    device = x.device.index
    fn = _avg_down_add_kernel(device, cin, cout, factor_t, factor_s, group_size, bias is not None)
    fn(
        x.data_ptr(),
        r.data_ptr(),
        bias.data_ptr() if bias is not None else 0,
        y.data_ptr(),
        out_positions,
        out_width,
        in_width,
        out_height * factor_s,
        pad_t,
        items,
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
    return y


class InterleaveBiasKernel:
    """``u [T, P, 2C]`` -> ``y [2T, P, C]``: output frame ``2t + h`` takes channels
    ``h*C .. h*C + C - 1`` of input frame ``t``, plus their bias (one rounding per element). One
    thread per 16-byte output vector."""

    def __init__(self, channels: int, has_bias: bool) -> None:
        self.cvec = channels // VEC
        self.has_bias = has_bias

    @cute.jit
    def wrapper(
        self,
        u_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        positions: Int32,
        items: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(u_ptr, bias_ptr, y_ptr, positions, items).launch(
            grid=[(items + THREADS - 1) // THREADS, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        u_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        positions: Int32,
        items: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        idx = Int32(bid) * THREADS + Int32(tid)
        if idx < items:
            uu = u32_view(u_ptr)
            ub = u32_view(bias_ptr)
            uy = u32_view(y_ptr)
            row_vecs = positions * self.cvec
            t_out = idx // row_vecs
            in_frame = idx - t_out * row_vecs
            pos = in_frame // self.cvec
            cv = in_frame - pos * self.cvec
            half = t_out % 2
            src = (
                (cutlass.Int64(t_out // 2) * positions + pos) * (2 * self.cvec)
                + half * self.cvec
                + cv
            )
            words = load_words(uu, src)
            if cutlass.const_expr(self.has_bias):
                vs = unpack_bf16(words)
                bs = unpack_bf16(load_words(ub, half * self.cvec + cv))
                words = []
                for i in cutlass.range_constexpr(VEC // 2):
                    words.append(pack2(vs[2 * i] + bs[2 * i], vs[2 * i + 1] + bs[2 * i + 1]))
            store_words(uy, cutlass.Int64(idx), words)


@functools.lru_cache(maxsize=None)
def _interleave_bias_kernel(device: int, channels: int, bias: bool):
    from cutlass.cute.runtime import make_ptr

    kernel = InterleaveBiasKernel(channels, bias)

    def compile_fn():
        ptrs = [make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16) for _ in range(3)]
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        return cute.compile(
            kernel.wrapper, *ptrs, Int32(1), Int32(1), stream, options="--enable-tvm-ffi"
        )

    name = f"interleave_bias[{channels},{int(bias)}]"
    return compile_cached("trtllm_wan_vae_frame_ops", name, compile_fn, [__file__])


def interleave_bias(u: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """``u [T, P, 2C]`` contiguous bf16 -> ``[2T, P, C]``: even frames from channels ``0:C``, odd
    from ``C:2C``, plus ``bias`` (``2C`` values) when given. ``C`` must be a multiple of 8."""
    if u.dim() != 3 or u.dtype != torch.bfloat16 or not u.is_contiguous() or u.shape[2] % 16:
        raise ValueError("u must be contiguous bf16 [T, P, 2C] with C a multiple of 8")
    frames, positions, c2 = u.shape
    if bias is not None:
        bias = bias.reshape(-1)
        if bias.numel() != c2 or bias.dtype != torch.bfloat16 or not bias.is_contiguous():
            raise ValueError("bias must be 2C contiguous bf16 values")
    y = torch.empty((2 * frames, positions, c2 // 2), dtype=u.dtype, device=u.device)
    items = 2 * frames * positions * (c2 // 2 // VEC)
    if items == 0:
        return y
    if items >= 2**31:
        raise ValueError("interleave_bias: too many output vectors for one launch")
    device = u.device.index
    _interleave_bias_kernel(device, c2 // 2, bias is not None)(
        u.data_ptr(),
        bias.data_ptr() if bias is not None else 0,
        y.data_ptr(),
        positions,
        items,
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
    return y


class UnpatchifyClampKernel:
    """``chunk [t, H, W, C*P*P]`` -> ``out[frame0 + f, h*P + j, w*P + i, c] = clamp(chunk[f, h,
    w, (c*P + i)*P + j], -1, 1)`` (the element mapping of ``wan_vae.unpatchify``). One thread
    per chunk pixel; each of its ``P`` output row segments (``P`` pixels of ``C`` channels) is
    written as packed bf16 pairs when ``C*P`` is even and ``C*P*P`` a multiple of 4 (8-byte
    reads), else element by element."""

    def __init__(self, channels: int, patch: int) -> None:
        self.C, self.P = channels, patch
        self.cpp = channels * patch * patch
        self.packed = (channels * patch) % 2 == 0 and self.cpp % 4 == 0

    @cute.jit
    def wrapper(
        self,
        chunk_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        height: Int32,
        width: Int32,
        frame0: Int32,
        items: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(chunk_ptr, out_ptr, height, width, frame0, items).launch(
            grid=[(items + THREADS - 1) // THREADS, 1, 1], block=[THREADS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        chunk_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        height: Int32,
        width: Int32,
        frame0: Int32,
        items: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        idx = Int32(bid) * THREADS + Int32(tid)
        if idx < items:
            C, P = self.C, self.P
            w = idx % width
            th = idx // width
            h = th % height
            f = th // height
            vals = []
            if cutlass.const_expr(self.packed):
                # 8-byte words of the chunk pixel, 4-byte words of the output row segments.
                src = cute.make_tensor(
                    cute.recast_ptr(chunk_ptr, dtype=Int32),
                    cute.make_layout((2, cutlass.Int64(1) << 40), stride=(1, 2)),
                )
                dst = cute.make_tensor(
                    cute.recast_ptr(out_ptr, dtype=Int32), cute.make_layout((1 << 40,))
                )
                for q in cutlass.range_constexpr(self.cpp // 4):
                    pair = src[(None, cutlass.Int64(idx) * (self.cpp // 4) + q)].load()
                    vals.extend(unpack_bf16([pair[0], pair[1]]))
            else:
                src = cute.make_tensor(chunk_ptr, cute.make_layout((1 << 40,)))
                dst = cute.make_tensor(out_ptr, cute.make_layout((1 << 40,)))
                for q in cutlass.range_constexpr(self.cpp):
                    vals.append(Float32(src[cutlass.Int64(idx) * self.cpp + q]))
            out_w = width * P
            for j in cutlass.range_constexpr(P):
                seg = []
                for i in cutlass.range_constexpr(P):
                    for c in cutlass.range_constexpr(C):
                        v = vals[(c * P + i) * P + j]
                        seg.append(cute.arch.fmin(cute.arch.fmax(v, Float32(-1.0)), Float32(1.0)))
                row = (cutlass.Int64(frame0 + f) * (height * P) + h * P + j) * out_w + w * P
                if cutlass.const_expr(self.packed):
                    base = row * C // 2  # C*P is even: every segment starts on a 4-byte word
                    for k in cutlass.range_constexpr(C * P // 2):
                        dst[base + k] = pack2(seg[2 * k], seg[2 * k + 1])
                else:
                    # The clamped values are bf16 already: the conversion is exact.
                    for k in cutlass.range_constexpr(C * P):
                        dst[row * C + k] = BFloat16(seg[k])


@functools.lru_cache(maxsize=None)
def _unpatchify_clamp_kernel(device: int, channels: int, patch: int):
    from cutlass.cute.runtime import make_ptr

    kernel = UnpatchifyClampKernel(channels, patch)

    def compile_fn():
        align = 8 if kernel.packed else 2
        ptrs = [
            make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=align) for _ in range(2)
        ]
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        return cute.compile(
            kernel.wrapper, *ptrs, *([Int32(1)] * 4), stream, options="--enable-tvm-ffi"
        )

    name = f"unpatchify_clamp[{channels},{patch}]"
    return compile_cached("trtllm_wan_vae_frame_ops", name, compile_fn, [__file__])


def unpatchify_clamp_into(chunk: torch.Tensor, out: torch.Tensor, frame0: int, patch: int) -> None:
    """Write ``clamp(unpatchify(chunk), -1, 1)`` into ``out[:, :, frame0:frame0 + t]``: ``chunk``
    ``[1, C*patch*patch, t, H, W]`` and ``out`` ``[1, C, T, H*patch, W*patch]``, both
    channels-last bf16. Same element mapping as :func:`wan_vae.unpatchify`."""
    cl = torch.channels_last_3d
    if chunk.dim() != 5 or out.dim() != 5 or chunk.shape[0] != 1 or out.shape[0] != 1:
        raise ValueError("chunk and out must be batch-1 5D tensors")
    if chunk.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("chunk and out must be bf16")
    if not chunk.is_contiguous(memory_format=cl) or not out.is_contiguous(memory_format=cl):
        raise ValueError("chunk and out must be channels-last 3D")
    _, cpp, frames, height, width = chunk.shape
    channels = out.shape[1]
    if cpp != channels * patch * patch or (height * patch, width * patch) != tuple(out.shape[3:]):
        raise ValueError("chunk must hold C*patch^2 channels at 1/patch of out's spatial size")
    if frame0 < 0 or frame0 + frames > out.shape[2]:
        raise ValueError("frame range")
    items = frames * height * width
    if items == 0:
        return
    device = chunk.device.index
    _unpatchify_clamp_kernel(device, channels, patch)(
        chunk.data_ptr(),
        out.data_ptr(),
        height,
        width,
        frame0,
        items,
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
