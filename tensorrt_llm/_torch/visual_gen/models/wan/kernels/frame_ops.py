# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WinoVae frame operations in CuTe DSL on batch-1 channels-last bf16 frames ``[T, P, C]``: the
down block's AvgDown3D shortcut add (:func:`avg_down_add`), the temporal upsample's frame
interleave with its bias (:func:`interleave_bias`) and the decoder output's unpatchify and
clamp (:func:`unpatchify_clamp_into`)."""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32

from .common import (
    CHANNELS_PER_VECTOR,
    THREADS,
    check_aligned,
    check_bias,
    compile_cached,
    current_stream,
    flat_1d,
    flat_view,
    grid_1d,
    linear_thread_index,
    load_words,
    pack2,
    pack_words,
    round_pair_bf16,
    store_words,
    u32_view,
    unpack_bf16,
    vec_view,
)


class AvgDownAddKernel:
    """``y = x (+ bias) + AvgDown3D(r)``, the residual down block's shortcut add.

    AvgDown3D groups the ``cin * factor`` values of an output pixel's window (input channel
    slowest, then the temporal tap, then the two spatial taps; ``factor = factor_t * factor_s^2``)
    into ``cout`` groups of ``group_size`` consecutive values and takes each group's mean. The
    kernel requires ``factor % group_size == 0``, so every group lies inside one input
    channel's window. A thread owns one output pixel and 8 input channels: it reads each window
    tap as one 16-byte vector (the residual is read exactly once) and produces the ``8 * factor
    / group_size`` output channels those inputs feed.

    Numerics, per output channel: the group's values summed in fp32 in window order (temporal
    pad taps add nothing), the mean ``sum * (1 / group_size)`` rounded to bf16, then ``x +
    bias`` in fp32 plus the mean, rounded once.
    """

    def __init__(
        self, cin: int, cout: int, factor_t: int, factor_s: int, group_size: int, has_bias: bool
    ) -> None:
        self.cin, self.cout = cin, cout
        self.factor_t, self.factor_s = factor_t, factor_s
        self.factor = factor_t * factor_s * factor_s
        self.group_size = group_size
        self.per_input = self.factor // group_size  # output channels per input channel
        self.has_bias = has_bias
        self.cvec = cin // CHANNELS_PER_VECTOR

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
        ).launch(grid=grid_1d(items), block=[THREADS, 1, 1], stream=stream)

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
        idx = linear_thread_index()
        if idx < items:
            x_words = u32_view(x_ptr)
            bias_words = u32_view(bias_ptr)
            y_words = u32_view(y_ptr)
            residual_vecs = vec_view(r_ptr)
            in_vec = idx % self.cvec
            pix = idx // self.cvec
            t_out = pix // out_positions
            position = pix - t_out * out_positions
            i_out = position // out_width
            j_out = position - i_out * out_width
            taps = self._load_window_taps(
                residual_vecs, in_vec, t_out, i_out, j_out, in_width, in_rows, pad_t
            )

            inv_group = Float32(1.0 / self.group_size)
            for w in cutlass.range_constexpr(self.per_input):
                out_vec = (
                    cutlass.Int64(pix) * (self.cout // CHANNELS_PER_VECTOR)
                    + in_vec * self.per_input
                    + w
                )
                xs = unpack_bf16(load_words(x_words, out_vec))
                bs = None
                if cutlass.const_expr(self.has_bias):
                    bs = unpack_bf16(load_words(bias_words, in_vec * self.per_input + w))
                means = self._group_means(taps, w, inv_group)
                store_words(y_words, out_vec, self._add_means(xs, bs, means))

    @cute.jit
    def _load_window_taps(
        self, residual_vecs, in_vec, t_out, i_out, j_out, in_width, in_rows, pad_t
    ) -> list:
        """The window taps of the thread's 8 input channels in window order (temporal, then the
        two spatial), as fp32 vectors; temporal pad taps are zero."""
        fs, ft = self.factor_s, self.factor_t
        taps = []
        for a in cutlass.range_constexpr(ft):
            t_in = t_out * ft + a - pad_t
            for b in cutlass.range_constexpr(fs):
                for d in cutlass.range_constexpr(fs):
                    row = (t_in * in_rows + i_out * fs + b) * in_width + j_out * fs + d
                    buf = cute.make_rmem_tensor((CHANNELS_PER_VECTOR,), BFloat16)
                    if t_in >= 0:
                        buf.store(
                            residual_vecs[(None, cutlass.Int64(row) * self.cvec + in_vec)].load()
                        )
                    else:
                        buf.fill(0.0)
                    taps.append(buf.load().to(Float32))
        return taps

    def _group_means(self, taps: list, w: int, inv_group: Float32) -> list:
        """The means of the 8 output channels of the thread's output vector ``w``."""
        means = []
        for k in range(CHANNELS_PER_VECTOR):
            # Output channel o, counted from the thread's first one, is group tap_group of the
            # window of input channel lane.
            o = w * CHANNELS_PER_VECTOR + k
            lane, tap_group = o // self.per_input, o % self.per_input
            acc = Float32(0.0)
            for g in range(self.group_size):
                acc = acc + taps[tap_group * self.group_size + g][lane]
            means.append(acc * inv_group)
        return means

    def _add_means(self, xs: list, bs: list | None, means: list) -> list:
        """``x (+ bias) + mean`` per channel, the mean rounded to bf16 first, as packed words."""
        words = []
        for i in range(CHANNELS_PER_VECTOR // 2):
            _, mean_lo, mean_hi = round_pair_bf16(means[2 * i], means[2 * i + 1])
            sums = []
            for h, mean in enumerate((mean_lo, mean_hi)):
                value = xs[2 * i + h]
                if self.has_bias:
                    value = value + bs[2 * i + h]
                sums.append(value + mean)
            words.append(pack2(sums[0], sums[1]))
        return words


@functools.lru_cache(maxsize=None)
def _avg_down_add_kernel(
    device: int, cin: int, cout: int, factor_t: int, factor_s: int, group_size: int, bias: bool
):
    from cutlass.cute.runtime import make_ptr

    kernel = AvgDownAddKernel(cin, cout, factor_t, factor_s, group_size, bias)

    def compile_fn():
        ptrs = [make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16) for _ in range(4)]
        return cute.compile(
            kernel.wrapper,
            *ptrs,
            *([Int32(1)] * 6),
            current_stream(device),
            options="--enable-tvm-ffi",
        )

    name = f"avg_down_add[{cin},{cout},{factor_t},{factor_s},{group_size},{int(bias)}]"
    return compile_cached("trtllm_wan_vae_frame_ops", name, compile_fn, [__file__])


def _check_avg_down_add(
    x: torch.Tensor,
    r: torch.Tensor,
    out_height: int,
    out_width: int,
    in_width: int,
    factor_t: int,
    factor_s: int,
    group_size: int,
    pad_t: int,
) -> None:
    """The operand rules of :func:`avg_down_add`."""
    for name, t in (("x", x), ("r", r)):
        if t.dtype != torch.bfloat16 or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous bf16")
    if x.dim() != 3 or r.dim() != 3:
        raise ValueError("x and r must be [T, P, C] frames")
    out_frames, out_positions, cout = x.shape
    in_frames, in_positions, cin = r.shape
    factor = factor_t * factor_s * factor_s
    if out_positions != out_height * out_width:
        raise ValueError("x positions must equal out_height * out_width")
    if in_positions != out_height * factor_s * in_width or in_width < out_width * factor_s:
        raise ValueError(
            "r positions must equal out_height * factor_s * in_width, with in_width at least "
            "out_width * factor_s"
        )
    if in_frames + pad_t != out_frames * factor_t:
        raise ValueError("frame counts do not match the temporal factor")
    if (
        cin * factor != cout * group_size
        or factor % group_size
        or cin % CHANNELS_PER_VECTOR
        or cout % CHANNELS_PER_VECTOR
    ):
        raise ValueError(
            "avg_down_add needs cin * factor == cout * group_size, group_size dividing factor "
            "and channel counts that are multiples of 8"
        )


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
    contiguous 16-byte aligned bf16 with ``P == out_height * factor_s * in_width``, ``pad_t``
    zero frames in front of ``r``; ``bias`` (contiguous ``[Cout]`` bf16) is the downsample
    conv's bias when that conv ran without it. ``group_size`` must divide ``factor_t *
    factor_s^2``. Returns ``y`` like ``x``."""
    _check_avg_down_add(
        x, r, out_height, out_width, in_width, factor_t, factor_s, group_size, pad_t
    )
    out_frames, out_positions, cout = x.shape
    cin = r.shape[2]
    check_bias(bias, cout, torch.bfloat16, "Cout")
    check_aligned(16, x=x, r=r, bias=bias)
    y = torch.empty_like(x)
    items = out_frames * out_positions * (cin // CHANNELS_PER_VECTOR)
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
        current_stream(device),
    )
    return y


class InterleaveBiasKernel:
    """``u [T, P, 2C]`` -> ``y [2T, P, C]``: output frame ``2t + h`` takes channels
    ``h*C .. h*C + C - 1`` of input frame ``t``, plus their bias (one rounding per element). One
    thread per 16-byte output vector."""

    def __init__(self, channels: int, has_bias: bool) -> None:
        self.cvec = channels // CHANNELS_PER_VECTOR
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
            grid=grid_1d(items), block=[THREADS, 1, 1], stream=stream
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
        idx = linear_thread_index()
        if idx < items:
            u_words = u32_view(u_ptr)
            bias_words = u32_view(bias_ptr)
            y_words = u32_view(y_ptr)
            frame_vecs = positions * self.cvec
            t_out = idx // frame_vecs
            vec_in_frame = idx - t_out * frame_vecs
            pos = vec_in_frame // self.cvec
            channel_vec = vec_in_frame - pos * self.cvec
            half = t_out % 2
            src = (
                (cutlass.Int64(t_out // 2) * positions + pos) * (2 * self.cvec)
                + half * self.cvec
                + channel_vec
            )
            words = load_words(u_words, src)
            if cutlass.const_expr(self.has_bias):
                vs = unpack_bf16(words)
                bs = unpack_bf16(load_words(bias_words, half * self.cvec + channel_vec))
                words = pack_words([vs[k] + bs[k] for k in range(CHANNELS_PER_VECTOR)])
            store_words(y_words, cutlass.Int64(idx), words)


@functools.lru_cache(maxsize=None)
def _interleave_bias_kernel(device: int, channels: int, bias: bool):
    from cutlass.cute.runtime import make_ptr

    kernel = InterleaveBiasKernel(channels, bias)

    def compile_fn():
        ptrs = [make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16) for _ in range(3)]
        return cute.compile(
            kernel.wrapper,
            *ptrs,
            Int32(1),
            Int32(1),
            current_stream(device),
            options="--enable-tvm-ffi",
        )

    name = f"interleave_bias[{channels},{int(bias)}]"
    return compile_cached("trtllm_wan_vae_frame_ops", name, compile_fn, [__file__])


def interleave_bias(u: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """``u [T, P, 2C]`` contiguous 16-byte aligned bf16 -> ``[2T, P, C]``: even frames from
    channels ``0:C``, odd from ``C:2C``, plus ``bias`` (contiguous ``[2C]`` bf16) when given.
    ``C`` must be a multiple of 8."""
    if u.dim() != 3 or u.dtype != torch.bfloat16 or not u.is_contiguous() or u.shape[2] % 16:
        raise ValueError("u must be contiguous bf16 [T, P, 2C] with C a multiple of 8")
    frames, positions, c2 = u.shape
    check_bias(bias, c2, torch.bfloat16, "2C")
    check_aligned(16, u=u, bias=bias)
    y = torch.empty((2 * frames, positions, c2 // 2), dtype=u.dtype, device=u.device)
    items = 2 * frames * positions * (c2 // 2 // CHANNELS_PER_VECTOR)
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
        current_stream(device),
    )
    return y


def _unpatchify_packed(channels: int, patch: int) -> bool:
    """Whether the unpatchify kernel moves packed bf16 pairs: ``C*P`` even and ``C*P*P`` a
    multiple of 4 (8-byte reads, 4-byte writes)."""
    return (channels * patch) % 2 == 0 and (channels * patch * patch) % 4 == 0


class UnpatchifyClampKernel:
    """``chunk [t, H, W, C*P*P]`` -> ``out[frame0 + f, h*P + j, w*P + i, c] = clamp(chunk[f, h,
    w, (c*P + i)*P + j], -1, 1)`` (the element mapping of ``wan_vae.unpatchify``). One thread
    per chunk pixel; each of its ``P`` output row segments (``P`` pixels of ``C`` channels) is
    written as packed bf16 pairs when :func:`_unpatchify_packed` holds, else element by
    element."""

    def __init__(self, channels: int, patch: int) -> None:
        self.channels, self.patch = channels, patch
        self.values_per_pixel = channels * patch * patch
        self.packed = _unpatchify_packed(channels, patch)

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
            grid=grid_1d(items), block=[THREADS, 1, 1], stream=stream
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
        idx = linear_thread_index()
        if idx < items:
            channels, patch = self.channels, self.patch
            w = idx % width
            th = idx // width
            h = th % height
            f = th // height
            vals = self._load_pixel(chunk_ptr, idx)
            dst = self._out_view(out_ptr)
            out_w = width * patch
            for j in cutlass.range_constexpr(patch):
                seg = []
                for i in cutlass.range_constexpr(patch):
                    for c in cutlass.range_constexpr(channels):
                        v = vals[(c * patch + i) * patch + j]
                        seg.append(cute.arch.fmin(cute.arch.fmax(v, Float32(-1.0)), Float32(1.0)))
                row = (
                    cutlass.Int64(frame0 + f) * (height * patch) + h * patch + j
                ) * out_w + w * patch
                self._store_row_segment(dst, row, seg)

    def _load_pixel(self, chunk_ptr: cute.Pointer, idx) -> list:
        """The ``C*P*P`` values of chunk pixel ``idx`` as floats, in channel order."""
        vals = []
        if self.packed:
            words = self.values_per_pixel // 4  # 8-byte words per pixel
            src = flat_view(cute.recast_ptr(chunk_ptr, dtype=Int32), 2)
            for q in range(words):
                pair = src[(None, cutlass.Int64(idx) * words + q)].load()
                vals.extend(unpack_bf16([pair[0], pair[1]]))
        else:
            src = flat_1d(chunk_ptr)
            for q in range(self.values_per_pixel):
                vals.append(Float32(src[cutlass.Int64(idx) * self.values_per_pixel + q]))
        return vals

    def _out_view(self, out_ptr: cute.Pointer) -> cute.Tensor:
        """The output as 4-byte words (packed) or bf16 elements."""
        if self.packed:
            return flat_1d(cute.recast_ptr(out_ptr, dtype=Int32))
        return flat_1d(out_ptr)

    def _store_row_segment(self, dst: cute.Tensor, row, seg: list) -> None:
        """Store the ``P * C`` clamped values of one output row segment starting at pixel
        ``row``."""
        channels, patch = self.channels, self.patch
        if self.packed:
            base = row * channels // 2  # C*P is even: every segment starts on a 4-byte word
            for k in range(channels * patch // 2):
                dst[base + k] = pack2(seg[2 * k], seg[2 * k + 1])
        else:
            # The clamped values are bf16 already: the conversion is exact.
            for k in range(channels * patch):
                dst[row * channels + k] = BFloat16(seg[k])


@functools.lru_cache(maxsize=None)
def _unpatchify_clamp_kernel(device: int, channels: int, patch: int):
    from cutlass.cute.runtime import make_ptr

    kernel = UnpatchifyClampKernel(channels, patch)

    def compile_fn():
        align = 8 if kernel.packed else 2
        ptrs = [
            make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=align) for _ in range(2)
        ]
        return cute.compile(
            kernel.wrapper,
            *ptrs,
            *([Int32(1)] * 4),
            current_stream(device),
            options="--enable-tvm-ffi",
        )

    name = f"unpatchify_clamp[{channels},{patch}]"
    return compile_cached("trtllm_wan_vae_frame_ops", name, compile_fn, [__file__])


def unpatchify_clamp_into(chunk: torch.Tensor, out: torch.Tensor, frame0: int, patch: int) -> None:
    """Write ``clamp(unpatchify(chunk), -1, 1)`` into ``out[:, :, frame0:frame0 + t]``: ``chunk``
    ``[1, C*patch*patch, t, H, W]`` and ``out`` ``[1, C, T, H*patch, W*patch]``, both
    channels-last bf16, 8-byte aligned when :func:`_unpatchify_packed` holds. Same element
    mapping as :func:`wan_vae.unpatchify`."""
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
        raise ValueError(
            f"frames [{frame0}, {frame0 + frames}) of chunk do not fit in out's {out.shape[2]}"
        )
    check_aligned(8 if _unpatchify_packed(channels, patch) else 2, chunk=chunk, out=out)
    items = frames * height * width
    if items == 0:
        return
    if items >= 2**31:
        raise ValueError("unpatchify_clamp_into: too many chunk pixels for one launch")
    device = chunk.device.index
    _unpatchify_clamp_kernel(device, channels, patch)(
        chunk.data_ptr(),
        out.data_ptr(),
        height,
        width,
        frame0,
        items,
        current_stream(device),
    )
