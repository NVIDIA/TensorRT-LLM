# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WinoVae: optimized Wan VAE.

``WinoVae`` mirrors ``WanVAE`` module for module and loads the same state dict. On CUDA it
runs the fused norm kernels, temporal and spatial Winograd convolutions, frame buffers for
the causal history and the fused frame operations, in bf16 on CUDA; where a kernel cannot
run (no CUDA toolkit for the runtime-compiled kernels, an unsupported shape or batch) the
module falls back to the eager computation of ``WanVAE``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution

from .cudnn_conv import conv_frontend, cudnn_frontend_available
from .jit_norm import FrameGeometry
from .wan_vae import (
    CACHE_T,
    DupUp3D,
    WanAttentionBlock,
    WanCausalConv3d,
    WanDecoder3d,
    WanEncoder3d,
    WanMidBlock,
    WanResample,
    WanResidualBlock,
    WanResidualDownBlock,
    WanResidualUpBlock,
    WanRMSNorm,
    WanUpBlock,
    WanVAE,
    WanVAEConfig,
    _causal_conv_with_cache,
    _channels_last_2d_if_needed,
    _channels_last_3d_if_needed,
    _decode_chunk_slices,
    _to_device_if_needed,
    patchify,
    unpatchify,
)
from .wino_vae_kernels import (
    add_rms_norm_silu_cuda,
    apply_in_bias,
    avg_down_add_cuda,
    can_use_cuda_rms_norm_silu,
    can_use_cuda_upsample_gather,
    cuda_frame_ops_available,
    cuda_op_available,
    cuda_wino_spatial_available,
    frames_to_video,
    frames_view,
    images_to_pairs,
    interleave_bias_cuda,
    pack_upsample_weights,
    pairs_to_images,
    rms_norm_silu_cuda,
    sm100_fused_eligible,
    sm100_fused_filters,
    sm100_spatial_eligible,
    sm100_spatial_filters,
    sm100_wino_available,
    unpatchify_clamp_available,
    unpatchify_clamp_into,
    upsample_gather_cuda,
    wino_combine_cuda,
    wino_norm_silu_cuda,
    wino_spatial_conv_cuda,
    wino_spatial_conv_fused,
    wino_spatial_conv_sm100,
    wino_spatial_conv_taps,
    winograd_spatial_taps_weights,
    winograd_spatial_weights,
    winograd_weights,
)
from .wino_vae_shard import DirectFrames, Shard, push_border

# One slot of the causal-conv cache list threaded through a chunked encode / decode: the
# conv's history frames ``[B, C, 0..2, H, W]``; the conv itself when it keeps them in its
# frame buffer (``_FrameBuffer.history`` hands them over); the ``"Rep"``
# sentinel of the resampler's time conv; None before the first chunk.
CacheEntry: TypeAlias = "torch.Tensor | WinoVaeCausalConv3d | Literal['Rep'] | None"


@dataclass(frozen=True)
class WinoVaeOptions:
    """Run-time settings of a WinoVae model.

    Frozen: the options are fixed for the model's lifetime. Build another model (with
    ``dataclasses.replace`` for a variant) to run with different ones."""

    # Pixel frames per encoder pass after the first one-frame pass. Must be a multiple of 4
    # so every pass ends at the same temporal-downsampler parity. Larger passes give the
    # convs more rows and fewer launches at the price of peak memory; from 8 frames up the
    # deepest levels see frame pairs and take the Winograd pair path.
    encoder_chunk_frames: int = 24
    # Chunks a conv's frame buffer holds after its two causal-history frames before it wraps
    # (the history moves to the front: a copy of two frames). Fewer chunks: less memory, more
    # wraps. ``decode``'s ``temporal_chunk_size`` and ``encoder_chunk_frames`` move peak
    # memory further.
    frame_buffer_chunks: int = 4
    # Spatially sharded runs: how the halo lines reach the neighbouring ranks. "direct": the
    # norm kernels store the border lines straight into the neighbours' buffers (symmetric
    # memory over NVLink) and one barrier per exchange replaces the pack copy, the NCCL
    # send/receive and the unpack copy; "nccl": batched point-to-point sends; "auto":
    # "direct" where the ranks can map each other's memory, else "nccl".
    halo_exchange: str = "auto"

    def __post_init__(self) -> None:
        if self.halo_exchange not in ("auto", "nccl", "direct"):
            raise ValueError(
                f"halo_exchange must be 'auto', 'nccl' or 'direct', got {self.halo_exchange!r}"
            )
        if self.frame_buffer_chunks < 1:
            raise ValueError(
                f"frame_buffer_chunks must be at least 1, got {self.frame_buffer_chunks!r}"
            )
        if self.encoder_chunk_frames < 4 or self.encoder_chunk_frames % 4:
            raise ValueError(
                f"encoder_chunk_frames must be a positive multiple of 4, got {self.encoder_chunk_frames!r}"
            )

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "WinoVaeOptions":
        valid = {field.name for field in fields(cls)}
        unknown = set(values) - valid
        if unknown:
            raise ValueError(f"Unknown WinoVaeOptions: {sorted(unknown)}; valid: {sorted(valid)}")
        return cls(**values)


# The decoder's narrow conv_out (256 -> 12 channels) runs on at most this many input frames
# per cuDNN call (the outputs are concatenated): beyond three frames cuDNN picks a much
# slower kernel for that shape.
_CONV_OUT_MAX_IN_FRAMES = 3
# Storage type of the temporal Winograd inputs and conv products between the norm kernels:
# three more mantissa bits than bf16 for the transformed values (required by F(4,3)).
_WINOGRAD_DTYPE = torch.float16
# Spatial Winograd tiles of the portable path: F(4x4, 3x3) for frame pairs, F(2x2, 3x3) for
# single frames with a causal history (the three temporal taps summed in the GEMM).
_SPATIAL_TILE = 4
_SPATIAL_SINGLE_TILE = 2


def temporal_order_for(frames: int) -> int:
    """The temporal Winograd order of a call with ``frames`` output frames: F(4,3) on the
    points (0, 1, -1, 2, -1/2, inf) for a multiple of four frames, else F(2,3)."""
    return 4 if frames % 4 == 0 else 2


def _benchmark_convs(*, encoder: bool, frames: int) -> bool:
    """cuDNN autotuning for the encoder and single-frame (image) decodes; video decodes keep
    cuDNN's heuristics."""
    return encoder or frames == 1


@dataclass
class PendingAdd:
    """``(a + bias) + b`` not yet computed: the next fused norm adds it while reading.

    Exactly one source holds the frames: ``a`` (a tensor) or ``wino`` (the temporal Winograd
    combine of the ``order + 2`` conv2d results, each ``[groups, C, H, W]`` channels-last, bf16
    or fp16: frame ``g*order + j`` is ``sum_k AT[j][k] m_k``, :data:`TEMPORAL_WINO`). ``bias``
    is a per-channel convolution bias the conv did not apply; ``b`` is a residual; either may
    be absent.
    """

    a: torch.Tensor | None
    b: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    wino: list[torch.Tensor] | None = None
    # Frame layout of the products when they carry halo lines (sharded decode).
    geometry: FrameGeometry | None = None

    def __post_init__(self) -> None:
        if (self.a is None) == (self.wino is None):
            raise ValueError("PendingAdd needs exactly one of a and wino")

    @property
    def shape(self) -> torch.Size:
        """The logical ``[B, C, T, H, W]`` of the frames once added."""
        src = self.a if self.a is not None else self.b
        batch = src.shape[0] if src is not None else 1
        channels = (src if src is not None else self.wino[0]).shape[1]
        return torch.Size((batch, channels, self.frames, *self.spatial))

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the frames once added (the combine kernel writes bf16)."""
        src = self.a if self.a is not None else self.b
        return src.dtype if src is not None else torch.bfloat16

    @property
    def device(self) -> torch.device:
        return (self.a if self.a is not None else self.wino[0]).device

    @property
    def frames(self) -> int:
        if self.a is not None:
            return self.a.shape[2]
        return (len(self.wino) - 2) * self.wino[0].shape[0]

    @property
    def spatial(self) -> tuple[int, int]:
        """The logical frame size (H, W)."""
        if self.geometry is not None:
            return self.geometry.height, self.geometry.width
        src = self.a if self.a is not None else self.wino[0]
        return src.shape[-2], src.shape[-1]

    def materialize(self) -> torch.Tensor:
        """The frames as a tensor: one combine kernel pass, or the eager adds of a tensor
        source the kernel does not take (batched, or the kernels are unavailable). Winograd
        products only exist when the kernels ran, so they always take the kernel."""
        if self.can_fuse():
            h, w = self.spatial
            return frames_to_video(self._combine(0, 0), h, w)
        s = apply_in_bias(self.a, self.bias)
        if self.b is not None:
            s = s + self.b
        return _channels_last_3d_if_needed(s)

    def _combine(self, pad_h: int, pad_w: int) -> torch.Tensor:
        h, w = self.spatial
        x_f = frames_view(_channels_last_3d_if_needed(self.a)) if self.a is not None else None
        m_f = [images_to_pairs(mk) for mk in self.wino] if self.wino is not None else None
        r_f = frames_view(_channels_last_3d_if_needed(self.b)) if self.b is not None else None
        return wino_combine_cuda(
            x_f,
            m_f,
            self.bias,
            r_f,
            self.frames,
            h,
            w,
            pad_h,
            pad_w,
            products_geometry=self.geometry,
        )

    def can_fuse(self) -> bool:
        """Whether the CUDA kernels can consume this pending add directly (batch 1, the
        runtime-compiled kernels available): the consumer norm adds it while reading and
        ``materialize`` uses the combine kernel."""
        return (
            cuda_op_available() and self.shape[0] == 1 and (self.b is None or self.b.shape[0] == 1)
        )

    def materialize_padded_2d(self) -> torch.Tensor:
        """The frames with one zero row and column appended, as ``[T, C, H+1, W+1]`` channels-last
        (the downsampler's input), in one pass."""
        src = self.a if self.a is not None else self.wino[0]
        h, w = self.spatial
        c = src.shape[1]
        y = self._combine(1, 1)  # [T, (H+1)*(W+1), C]
        return y.view(self.frames, h + 1, w + 1, c).permute(0, 3, 1, 2)


@dataclass
class WinogradInputs:
    """The ``order + 2`` transformed inputs of a temporal Winograd conv, each
    ``[groups, C, H, W]`` channels-last (bf16 or fp16), plus the buffer they are views of as
    ``[order + 2, groups, H*W, C]``. With ``geometry`` the frames carry halo lines (physical
    ``rows x cols``; the logical ``H x W`` sits inside)."""

    d: list[torch.Tensor]
    stacked: torch.Tensor
    geometry: FrameGeometry | None = None

    def physical(self) -> torch.Tensor:
        """The buffer as ``[comps, groups, rows, cols, C]``."""
        comps, groups, _, c = self.stacked.shape
        if self.geometry is None:
            _, _, h, w = self.d[0].shape
            return self.stacked.view(comps, groups, h, w, c)
        return self.stacked.view(comps, groups, self.geometry.rows, self.geometry.cols, c)


def _cache_frames(entry: CacheEntry) -> torch.Tensor | Literal["Rep"] | None:
    """A ``feat_cache`` entry as the history-frame tensor the eager and Winograd paths read:
    a conv that left itself there (frame-buffer path) hands over its two history frames."""
    return entry.frame_buffer.history() if isinstance(entry, WinoVaeCausalConv3d) else entry


def _frames_physical(window: torch.Tensor) -> torch.Tensor:
    """A channels-last ``[1, C, T, rows, cols]`` window as ``[T, rows, cols, C]``."""
    return window.permute(0, 2, 3, 4, 1)[0]


def _copy_into(
    out: torch.Tensor | None, value: torch.Tensor, geometry: FrameGeometry | None = None
) -> torch.Tensor:
    """Copy ``value`` into ``out`` when a destination was requested (into the logical
    positions when ``out`` carries halo lines)."""
    if out is not None and geometry is not None:
        g = geometry
        out[..., g.row0 : g.row0 + g.height, g.col0 : g.col0 + g.width].copy_(value)
        if g.push is not None:
            push_border(g.push, _frames_physical(out), g)
        return out
    if out is None:
        return value
    out.copy_(value)
    return out


def _as_tensor(x: "torch.Tensor | PendingAdd") -> torch.Tensor:
    return x.materialize() if isinstance(x, PendingAdd) else x


class WinoVaeRMSNorm(WanRMSNorm):
    """``WanRMSNorm`` with the fused CUDA entry points: ``norm_silu`` (rows) and
    ``norm_silu_winograd`` (the temporal Winograd inputs of the conv that follows)."""

    def __init__(
        self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False
    ) -> None:
        super().__init__(dim, channel_first, images, bias)
        self.shard: Shard | None = None

    def norm_silu(
        self,
        x: "torch.Tensor | PendingAdd",
        *,
        out: torch.Tensor | None = None,
        out_geometry: FrameGeometry | None = None,
        return_sum: bool = False,
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        """``SiLU(norm(s))`` with ``s`` = ``x`` (a tensor, or a pending add computed while
        reading where the kernel supports it). ``out``: optional channels-last destination (a
        conv frame-buffer window; with ``out_geometry`` its frames carry halo lines and the
        logical positions are written). ``return_sum``: return ``(s, SiLU(norm(s)))``."""
        bias = self.bias if isinstance(self.bias, torch.Tensor) else None
        in_bias = None
        if isinstance(x, PendingAdd):
            if (
                x.a is not None
                and x.b is not None
                and self.channel_first
                and can_use_cuda_rms_norm_silu(x.a)
                and can_use_cuda_rms_norm_silu(x.b)
            ):
                s, y = add_rms_norm_silu_cuda(
                    x.a,
                    x.b,
                    self.gamma,
                    bias,
                    self.scale,
                    in_bias=x.bias,
                    out=out,
                    out_geometry=out_geometry,
                )
                return (s, y) if return_sum else y
            if x.a is not None and x.b is None and not return_sum:
                x, in_bias = x.a, x.bias  # the kernel adds the deferred bias while reading
            else:
                x = x.materialize()
        if self.channel_first and can_use_cuda_rms_norm_silu(x):
            y = rms_norm_silu_cuda(
                x,
                self.gamma,
                bias,
                self.scale,
                in_bias=in_bias,
                out=out,
                out_geometry=out_geometry,
            )
        else:
            y = _copy_into(out, F.silu(self.forward(apply_in_bias(x, in_bias))), out_geometry)
        return (x, y) if return_sum else y

    def norm_silu_winograd(
        self,
        x: "torch.Tensor | PendingAdd",
        cache: torch.Tensor | None,
        frames: int,
        inputs_geometry: FrameGeometry | None = None,
        inputs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, WinogradInputs]:
        """Norm + SiLU that writes the Winograd inputs for the conv that follows.

        ``cache`` holds the last 0..2 normalized frames of the previous chunk as
        ``[1, C, n, H, W]``. Returns ``(sum or None, new cache [1, C, 2, H, W], inputs)``.
        ``inputs_geometry``: write the Winograd inputs into frames with halo lines (sharded decode);
        ``inputs``: a preallocated destination for them."""
        pending = x if isinstance(x, PendingAdd) else PendingAdd(x)
        h, w = pending.spatial
        bias = self.bias if isinstance(self.bias, torch.Tensor) else None
        s, y_tail, d = wino_norm_silu_cuda(
            frames_view(pending.a) if pending.a is not None else None,
            [images_to_pairs(mk) for mk in pending.wino] if pending.wino is not None else None,
            frames_view(pending.b) if pending.b is not None else None,
            pending.bias,
            frames_view(cache) if cache is not None and cache.shape[2] > 0 else None,
            self.gamma,
            bias,
            self.scale,
            frames,
            order=temporal_order_for(frames),
            fp16_inputs=_WINOGRAD_DTYPE == torch.float16,
            products_geometry=pending.geometry,
            inputs_geometry=inputs_geometry,
            inputs=inputs,
        )
        if inputs_geometry is None:
            inputs = WinogradInputs([pairs_to_images(dk, h, w) for dk in d], d)
        else:
            g = inputs_geometry
            inputs = WinogradInputs([pairs_to_images(dk, g.rows, g.cols) for dk in d], d, g)
        return (
            (frames_to_video(s, h, w) if s is not None else None),
            frames_to_video(y_tail, h, w),
            inputs,
        )


# Spatial Winograd for the temporal Winograd components, by kernel: the filter transform of a
# conv (built once per weight version) and the conv of the stacked components.
_SPATIAL_KERNELS = {
    # Blackwell: the fused F(2,3)-horizontal / direct-vertical 256-output-channel conv.
    "fused": (
        lambda conv, order: sm100_fused_filters(conv._winograd_weights(order)),
        wino_spatial_conv_fused,
    ),
    # Blackwell: the F(4x4, 3x3) CuTe DSL pipeline.
    "sm100": (
        lambda conv, order: sm100_spatial_filters(conv._winograd_weights(order)),
        wino_spatial_conv_sm100,
    ),
    # Any GPU: CUDA transform kernels and cuBLAS batched GEMMs.
    "cublas": (
        lambda conv, order: winograd_spatial_weights(conv.weight, _SPATIAL_TILE, order),
        wino_spatial_conv_cuda,
    ),
}


class _FrameBuffer:
    """A conv's frame buffer for the causal history: channels-last ``[1, C, depth, rows, cols]``
    frames (``rows x cols`` = the logical frame plus the halo lines under sharding). Frames
    ``[pos-2, pos)`` are the history (zeros before a clip's first chunk); the producer writes
    a chunk at ``[pos, pos+T)`` and the conv reads ``[pos-2, pos+T)`` with no copy. When a
    chunk would run off the end, the two history frames move to the front (one copy). The
    buffer outlives a clip: a new clip starts at the front over zeroed history. ``depth`` is
    ``2 + chunks * T`` for the chunk length the buffer was allocated for."""

    def __init__(self, channels: int, chunks: int = 4) -> None:
        self.channels = channels
        self.chunks = chunks
        self.shard: Shard | None = None
        self.tensor: torch.Tensor | None = None
        self.pos = 0  # 0: a new clip, no history yet
        # Frame layout of the current window (halo lines and the direct push under sharding).
        self.geometry: FrameGeometry | None = None
        self._key: tuple | None = None
        # The buffer as a symmetric-memory allocation (direct halo exchange).
        self._direct: DirectFrames | None = None

    def new_clip(self) -> None:
        self.pos = 0

    def window(
        self, frames: int, dtype: torch.dtype, device: torch.device, spatial: tuple[int, int]
    ) -> torch.Tensor:
        """The writable ``[1, C, frames, rows, cols]`` view for the next chunk of a logical
        ``spatial`` = (H, W) frame size."""
        h, w = spatial
        shard = self.shard
        geo = shard.geometry(h, w) if shard is not None else None
        size = (geo.rows, geo.cols) if geo is not None else (h, w)
        # The layout in terms every rank agrees on (an unequal split can change one rank's
        # extent and not another's, and the direct allocation is collective).
        key = (dtype, shard.frame_key(h, w) if shard is not None else (h, w))
        if self.tensor is None or self._key != key:
            self._replace(frames, dtype, device, size, geo, keep_history=False)
            self._key = key
        elif self.pos == 0:
            if self.tensor.shape[2] < 2 + frames:
                self._replace(frames, dtype, device, size, geo, keep_history=False)
            else:
                self.tensor[:, :, :2].zero_()
                self.pos = 2
        elif self.pos + frames > self.tensor.shape[2]:
            self._wrap(frames, dtype, device, size, geo)
        if self._direct is not None:
            geo = replace(geo, push=self._direct.push(self.pos, frames))
        self.geometry = geo
        return self.tensor[:, :, self.pos : self.pos + frames]

    def _wrap(
        self,
        frames: int,
        dtype: torch.dtype,
        device: torch.device,
        size: tuple[int, int],
        geo: FrameGeometry | None,
    ) -> None:
        """Move the two history frames to the front. Into a new buffer when this chunk is
        longer than the buffer holds, or when the neighbours push into this one: they run up
        to one exchange ahead and store their border lines into the new window ``[2, 2 +
        frames)`` as soon as they reach it, which must not overtake the history copy."""
        too_short = self.tensor.shape[2] < 2 + frames
        pushed_into = self._direct is not None and self.pos - 2 < 2 + frames
        if too_short or pushed_into:
            self._replace(frames, dtype, device, size, geo, keep_history=True)
            return
        hist = self.tensor[:, :, self.pos - 2 : self.pos]
        if self.pos < 4:  # source and destination overlap (one frame per chunk)
            hist = hist.clone()
        self.tensor[:, :, :2].copy_(hist)
        self.pos = 2

    def _replace(
        self,
        frames: int,
        dtype: torch.dtype,
        device: torch.device,
        size: tuple[int, int],
        geo: FrameGeometry | None,
        *,
        keep_history: bool,
    ) -> None:
        """A new zeroed buffer for ``frames``-frame chunks (zeros: the history of a new clip
        and, sharded, the outer halo of the edge ranks); with ``keep_history`` the current
        two history frames are copied over first and the old buffer is released after it."""
        depth = 2 + self.chunks * frames
        old, old_direct = self.tensor, self._direct
        if self.shard is not None and self.shard.direct:
            self._direct = self.shard.direct_frames(1, depth, geo, 1, self.channels, dtype)
            tensor = self._direct.frames[0].unsqueeze(0).permute(0, 4, 1, 2, 3)
        else:
            self._direct = None
            tensor = torch.empty(
                1,
                self.channels,
                depth,
                *size,
                dtype=dtype,
                device=device,
                memory_format=torch.channels_last_3d,
            ).zero_()
        if keep_history:
            tensor[:, :, :2].copy_(old[:, :, self.pos - 2 : self.pos])
        if old_direct is not None:
            self.shard.retire(old_direct)
        self.tensor, self.pos = tensor, 2

    def history(self) -> torch.Tensor | None:
        """The two history frames as a logical channels-last ``[1, C, 2, H, W]`` tensor (the
        cache entry the other conv paths read), or ``None`` before the first chunk."""
        if self.tensor is None or self.pos < 2:
            return None
        hist = self.tensor[:, :, self.pos - 2 : self.pos]
        geo = self.geometry
        if geo is not None:
            hist = hist[:, :, :, geo.row0 : geo.row0 + geo.height, geo.col0 : geo.col0 + geo.width]
        return hist.contiguous(memory_format=torch.channels_last_3d)

    def exchange(self, window: torch.Tensor) -> None:
        """Complete the halo lines of a just-written window: wait for the neighbours' direct
        stores, or trade the border lines over NCCL."""
        if self.shard is None:
            return
        if self._direct is not None:
            self._direct.barrier()
        else:
            self.shard.exchange(_frames_physical(window), self.geometry)


class WinoVaeCausalConv3d(WanCausalConv3d):
    """``WanCausalConv3d`` with the frame-buffer, temporal and spatial Winograd paths."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)
        # The Winograd, frame-buffer and single-frame paths take dense unit-stride convs.
        self._unit_stride = (
            self.stride == (1, 1, 1) and self.dilation == (1, 1, 1) and self.groups == 1
        )
        # When set, forward() leaves the bias out; the caller hands it to the consumer kernel.
        self.defer_bias = False
        # When set, a stride-1 temporal conv runs on at most this many input frames per
        # cuDNN call and concatenates the output frames.
        self.max_in_frames: int | None = None
        # Tensors derived from the weight (transformed filters, the current-frame slice).
        self._derived: dict[tuple, tuple[tuple[int, int], Any]] = {}
        # Set by the residual block for its two convs: spatial Winograd where selected, and
        # the cuDNN graph API (per-shape plan choice) for the convs that stay direct. Every
        # other conv runs through torch's cuDNN path.
        self.spatial_winograd = False
        self.cudnn_frontend = False
        # The causal history of the frame-buffer path (see _FrameBuffer).
        self.frame_buffer = _FrameBuffer(in_channels)
        # Spatial sharding: this rank's slice and the halo exchanges (None: whole frames).
        self.shard = None

    def buffer_begin(
        self,
        cache_entry: CacheEntry,
        frames: int,
        like: "torch.Tensor | PendingAdd",
        spatial: tuple[int, int],
    ) -> torch.Tensor:
        """Start a frame-buffer chunk: a new clip on the first chunk, adopting a cache tensor
        left by the eager first-chunk path as the history, and return the writable window of
        a logical ``spatial`` = (H, W) frame."""
        buf = self.frame_buffer
        adopt = cache_entry if isinstance(cache_entry, torch.Tensor) else None
        if cache_entry is None or adopt is not None:
            buf.new_clip()
        window = buf.window(frames, like.dtype, like.device, spatial)
        if adopt is not None:
            n = adopt.shape[2]
            if self.shard is not None:
                adopt = self.shard.pad_frames(_channels_last_3d_if_needed(adopt))
            buf.tensor[:, :, 2 - n : 2].copy_(adopt)
        return window

    @property
    def shard(self) -> Shard | None:
        return self._shard

    @shard.setter
    def shard(self, shard: Shard | None) -> None:
        self._shard = shard
        self.frame_buffer.shard = shard

    def buffer_conv(self, frames: int) -> torch.Tensor:
        """Run the conv on the frame buffer's ``[pos-2, pos+frames)`` and advance; the next two
        history frames stay in place."""
        buf = self.frame_buffer.tensor
        pos = self.frame_buffer.pos
        bias = None if self.defer_bias else self.bias
        padding = self.padding if self.shard is None else self.shard.conv_padding(self.padding)
        if (
            frames == 1
            and self.shard is None
            and self.spatial_eligible(buf.shape[3], buf.shape[4], _SPATIAL_SINGLE_TILE)
        ):
            y = self._forward_spatial_taps(
                [frames_view(buf[:, :, pos - 2 : pos + 1])], 0, buf.shape[3], buf.shape[4], bias
            )
            self.frame_buffer.pos = pos + 1
            return y
        y = self._conv3d(buf[:, :, pos - 2 : pos + frames], bias, padding)
        self.frame_buffer.pos = pos + frames
        return _channels_last_3d_if_needed(y)

    def _conv3d(
        self, x: torch.Tensor, bias: torch.Tensor | None, padding: tuple[int, int, int]
    ) -> torch.Tensor:
        """The 3D conv of an already causally padded ``x``: in calls of at most
        ``max_in_frames`` input frames when set (concatenated), else through the cuDNN graph
        API with the plan selected for this shape when ``cudnn_frontend`` is set (and the
        frontend is installed, the call unbatched), else torch's cuDNN path."""
        kt = self.kernel_size[0]
        if (
            self.max_in_frames is not None
            and self.stride[0] == 1
            and x.shape[2] > self.max_in_frames >= kt
        ):
            per_call = self.max_in_frames - kt + 1
            parts = [
                F.conv3d(
                    x[:, :, t : t + per_call + kt - 1],
                    self.weight,
                    bias,
                    self.stride,
                    padding,
                    self.dilation,
                    self.groups,
                )
                for t in range(0, x.shape[2] - kt + 1, per_call)
            ]
            return _channels_last_3d_if_needed(torch.cat(parts, dim=2))
        if (
            self.cudnn_frontend
            and self.groups == 1
            and x.shape[0] == 1
            and cudnn_frontend_available()
        ):
            w = self.weight.detach()
            if not w.is_contiguous(memory_format=torch.channels_last_3d):
                w = w.contiguous(memory_format=torch.channels_last_3d)
            return conv_frontend(
                x.contiguous(memory_format=torch.channels_last_3d),
                w,
                bias,
                tuple(self.stride),
                tuple(padding),
                tuple(self.dilation),
            )
        return F.conv3d(x, self.weight, bias, self.stride, padding, self.dilation, self.groups)

    def buffer_eligible(self, x: "torch.Tensor | PendingAdd") -> bool:
        return x.shape[0] == 1 and self.kernel_size[0] == 3 and self._unit_stride

    def _derived_weight(self, key: tuple, build: Callable[[], Any]) -> Any:
        """A tensor derived from the weight, built on first use and rebuilt when the weight
        tensor changes identity or version."""
        stamp = (self.weight.data_ptr(), self.weight._version)
        cached = self._derived.get(key)
        if cached is None or cached[0] != stamp:
            cached = (stamp, build())
            self._derived[key] = cached
        return cached[1]

    def _spatial_winograd(
        self, kernel: str, order: int, stacked: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        """The ``order + 2`` components' 3x3 convs by spatial Winograd with ``kernel`` (see
        ``_SPATIAL_KERNELS``): ``[comps, pairs, h*w, Cin]`` -> ``[comps, pairs, h*w, Cout]``."""
        filters, conv = _SPATIAL_KERNELS[kernel]
        u = self._derived_weight((kernel, order), lambda: filters(self, order))
        return conv(stacked, u, h, w)

    def _winograd_weights(self, order: int) -> list[torch.Tensor]:
        """The ``order + 2`` temporal component filters, in the Winograd dtype."""
        return self._derived_weight(
            ("temporal", order), lambda: winograd_weights(self.weight, order, _WINOGRAD_DTYPE)
        )

    def winograd_eligible(self, frames: int, batch: int) -> bool:
        return (
            batch == 1
            and frames >= 2
            and frames % 2 == 0
            and self.kernel_size == (3, 3, 3)
            and self._unit_stride
        )

    def spatial_eligible(
        self, height: int, width: int, tile: int = _SPATIAL_TILE, kernel: str = "cublas"
    ) -> bool:
        """Whether spatial Winograd runs this conv on ``height x width`` frames with ``kernel``
        ("sm100" or the portable "cublas") and ``tile``."""
        if (
            not self.spatial_winograd
            or self.padding[1:] != (1, 1)
            or self.in_channels % 8
            or self.out_channels % 8
        ):
            return False
        return self._spatial_selected(height, width, tile, kernel)

    def _spatial_selected(self, height: int, width: int, tile: int, kernel: str) -> bool:
        """The layers spatial Winograd is used on: the sm100 F(4x4, 3x3) pipeline on convs with
        at least 512 input channels up to 360x640; the portable path with F(4x4, 3x3) from 512
        input channels up to 180x320, with F(2x2, 3x3) from 1024 input channels up to 90x160."""
        if kernel == "sm100":
            return self.in_channels >= 512 and height * width <= 360 * 640
        if tile == 4:
            return self.in_channels >= 512 and height * width <= 180 * 320
        return self.in_channels >= 1024 and height * width <= 90 * 160

    def _forward_spatial_taps(
        self,
        taps: list[torch.Tensor],
        zero_front: int,
        h: int,
        w: int,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """One output frame by spatial Winograd alone from the taps' input frames in their own
        buffers (``[n, H*W, C]`` each, oldest first, ``zero_front`` zero frames before them):
        the new frame alone (zero history) or the causal history and the new frame."""
        n_taps = zero_front + sum(t.shape[0] for t in taps)
        tile = _SPATIAL_TILE if n_taps == 1 else _SPATIAL_SINGLE_TILE
        u = self._derived_weight(
            ("taps", n_taps, tile),
            lambda: winograd_spatial_taps_weights(self.weight, tile, n_taps),
        )
        y = wino_spatial_conv_taps(taps, zero_front, u, h, w, bias)
        return frames_to_video(y.unsqueeze(0), h, w)

    def _spatial_kernel(self, h: int, w: int) -> str:
        """Which conv runs the Winograd inputs of a logical ``h x w`` frame: "fused" (the
        fused F(2,3)-horizontal / direct-vertical 256-channel conv), "sm100" (the Blackwell
        F(4x4) pipeline), "cublas" (the portable spatial Winograd) or "cudnn" (conv2d per
        Winograd input). On a shard the
        choice is made for the largest shard's frame, so every rank picks the same kernel and
        the same Winograd-input layout."""
        if self.shard is not None:
            if self.shard.axis == "h":
                h = max(self.shard.output_extents(h))
            else:
                w = max(self.shard.output_extents(w))
        sm100 = sm100_wino_available()
        cin, cout = self.in_channels, self.out_channels
        if self.spatial_winograd and sm100 and sm100_fused_eligible(cin, cout, h, w):
            return "fused"
        if (
            sm100
            and sm100_spatial_eligible(cin, cout, w)
            and self.spatial_eligible(h, w, kernel="sm100")
        ):
            return "sm100"
        if self.spatial_eligible(h, w):
            return "cublas"
        return "cudnn"

    def winograd_inputs_align(self, h: int, w: int, order: int) -> int:
        """Lines the physical extent of this conv's Winograd-input frames is a multiple of: the tile of
        the kernel that will read them (the fused 256-channel conv takes 8-row x 32-column
        tiles, the sm100 pipeline 4-wide columns), 1 otherwise."""
        kernel = self._spatial_kernel(h, w)
        if kernel == "fused":
            return 8 if self.shard.axis == "h" else 32
        if kernel == "sm100" and self.shard.axis == "w":
            return 4
        return 1

    def winograd_inputs_geometry(self, h: int, w: int, order: int) -> FrameGeometry | None:
        """The Winograd-input layout for a logical ``h x w`` frame under sharding: halo lines, with the
        physical extent rounded up to :meth:`winograd_inputs_align`. None without sharding."""
        if self.shard is None:
            return None
        return self.shard.geometry(h, w, self.winograd_inputs_align(h, w, order))

    def winograd_inputs(
        self, h: int, w: int, order: int, frames: int
    ) -> tuple[FrameGeometry | None, torch.Tensor | None, DirectFrames | None]:
        """What the norm kernel that feeds this conv writes its Winograd inputs into for a logical
        ``h x w`` frame: the Winograd-input layout (None without sharding) and, with direct halo
        exchange, the symmetric buffer ``[order + 2, frames / order, positions, C]`` to fill
        (the layout then carries the neighbours' halo lines as its push) and that buffer's
        handle for the barrier."""
        if self.shard is None:
            return None, None, None
        align = self.winograd_inputs_align(h, w, order)
        geo = self.shard.geometry(h, w, align)
        if not self.shard.direct:
            return geo, None, None
        comps, groups = order + 2, frames // order
        buf = self.shard.direct_inputs(comps, groups, geo, align, self.in_channels, _WINOGRAD_DTYPE)
        inputs = buf.frames.view(comps, groups, geo.frame, self.in_channels)
        return replace(geo, push=buf.push()), inputs, buf

    def forward_winograd(
        self, inputs: WinogradInputs
    ) -> tuple[list[torch.Tensor], FrameGeometry | None]:
        """The ``order + 2`` 3x3 convs of the transformed inputs (bias left to the consumer):
        cuDNN conv2d per input, or spatial Winograd on all of them at once where that is faster.
        Returns the products and their frame layout: the spatial Winograd kernels work on the
        physical frames (halo lines included), cuDNN emits the logical frame."""
        comps = len(inputs.d)
        order = comps - 2
        geo = inputs.geometry
        if geo is None:
            _, _, h, w = inputs.d[0].shape
            ph, pw = h, w
        else:
            h, w = geo.height, geo.width
            ph, pw = geo.rows, geo.cols
        kernel = self._spatial_kernel(h, w)
        if kernel in _SPATIAL_KERNELS:
            m = self._spatial_winograd(kernel, order, inputs.stacked, ph, pw)
            return [pairs_to_images(m[k], ph, pw) for k in range(comps)], geo
        g = self._winograd_weights(order)
        pad = (self.padding[1], self.padding[2])
        if geo is not None:
            # The halo lines supply the context on the split axis: no padding there, and the
            # conv emits the logical frame.
            pad = self.shard.conv_padding(pad)
        if self.cudnn_frontend and cudnn_frontend_available():
            # One stacked product buffer [comps, groups, H, W, Cout]: the consumer norm kernel
            # reads the components at one stride, and the graph API writes into the slices.
            groups = inputs.d[0].shape[0]
            out = torch.empty(
                (comps, groups, h, w, self.out_channels),
                dtype=inputs.d[0].dtype,
                device=inputs.d[0].device,
            )
            views = [
                out[k].permute(0, 3, 1, 2) for k in range(comps)
            ]  # [groups, Cout, H, W] channels-last
            for k in range(comps):
                conv_frontend(inputs.d[k], g[k], None, (1, 1), pad, (1, 1), out=views[k])
            return views, None
        return [F.conv2d(inputs.d[k], g[k], None, 1, pad) for k in range(comps)], None

    def forward(
        self,
        x: torch.Tensor,
        cache_x: torch.Tensor | None = None,
        *,
        spatial_padding: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """The causal conv of ``x`` after its cached history ``cache_x`` (None: zero history).
        A single frame takes a shortcut: a 2D conv on the current-frame taps when there is no
        history, spatial Winograd with the three taps summed in the GEMM when there is."""
        x = _channels_last_3d_if_needed(x)
        bias = None if self.defer_bias else self.bias
        if self.shard is not None and self.kernel_size[1:] != (1, 1) and spatial_padding is None:
            # The neighbours' border lines replace the zero padding on the split axis.
            x = self.shard.pad_frames(x)
            if cache_x is not None:
                cache_x = self.shard.pad_frames(
                    _channels_last_3d_if_needed(_to_device_if_needed(cache_x, x.device))
                )
            spatial_padding = self.shard.conv_padding(self.padding[1:])
        if x.shape[2] == 1:
            if cache_x is None and self._padding[4] > 0 and self.stride[0] == 1:
                return self._first_frame(x, bias, spatial_padding)
            if spatial_padding is None and self._single_frame_spatial_eligible(x, cache_x):
                return self._single_frame_spatial(x, cache_x, bias)
        # The history goes in front through a copy (torch.cat, as in WanCausalConv3d). The
        # large convs never come here: they read their history in place from a frame buffer
        # or take the temporal Winograd path. What does is small: the conv_in layers on the
        # model's input, the resamplers' time convs, and the eager path of batched calls. A
        # frame buffer would not save the copy there, writing the input into it is the copy.
        x = self._causal_input(x, cache_x)
        return self._conv3d(
            x, bias, self.padding if spatial_padding is None else (0, *spatial_padding)
        )

    def _first_frame(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None,
        spatial_padding: tuple[int, int] | None,
    ) -> torch.Tensor:
        """One frame with zero history: only the current-frame taps contribute (a 2D conv)."""
        w2 = self._derived_weight(
            ("current_frame",),
            lambda: self.weight[:, :, -1].contiguous(memory_format=torch.channels_last),
        )
        y = F.conv2d(
            x[:, :, 0],
            w2,
            bias,
            self.stride[1:],
            self.padding[1:] if spatial_padding is None else spatial_padding,
            self.dilation[1:],
            self.groups,
        )
        return _channels_last_3d_if_needed(y.unsqueeze(2))

    def _single_frame_spatial_eligible(self, x: torch.Tensor, cache_x: torch.Tensor | None) -> bool:
        return (
            self.kernel_size[0] == 3
            and self._padding[4] == 2
            and x.shape[0] == 1
            and self._unit_stride
            and (cache_x is None or cache_x.shape[2] <= 2)
            and self.spatial_eligible(x.shape[3], x.shape[4], _SPATIAL_SINGLE_TILE)
        )

    def _single_frame_spatial(
        self, x: torch.Tensor, cache_x: torch.Tensor | None, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """One frame after its history by spatial Winograd alone, the three taps summed inside
        the GEMM and the history read where the cache holds it."""
        taps = []
        if cache_x is not None:
            cache_x = _channels_last_3d_if_needed(_to_device_if_needed(cache_x, x.device))
            taps.append(frames_view(cache_x))
        taps.append(frames_view(x))
        history = 0 if cache_x is None else cache_x.shape[2]
        return self._forward_spatial_taps(taps, 2 - history, x.shape[3], x.shape[4], bias)


class WinoVaeResample(WanResample):
    """``WanResample`` with the fused upsample (2x2 conv on the low-res grid plus a phase
    gather), the deferred downsample bias and sharding."""

    _conv_cls = WinoVaeCausalConv3d

    def __init__(self, dim: int, mode: str, upsample_out_dim: int | None = None) -> None:
        super().__init__(dim, mode, upsample_out_dim)
        self._packed: tuple[int, int, torch.Tensor, torch.Tensor | None] | None = None
        # Downsample conv bias left out of the conv: the time conv absorbs it (downsample3d)
        # or the owning block's shortcut-add kernel adds it (downsample2d). Only a block that
        # takes ``pending_bias`` may enable the deferral.
        self.defer_bias_to_consumer = False
        self.pending_bias: torch.Tensor | None = None
        self._time_bias: tuple[int, int, int, int, torch.Tensor] | None = None
        self.shard: Shard | None = None

    def _packed_weights(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        conv = self.resample[1]
        key = (conv.weight.data_ptr(), conv.weight._version)
        if self._packed is None or self._packed[:2] != key:
            packed, packed_bias = pack_upsample_weights(conv.weight, conv.bias)
            self._packed = (key[0], key[1], packed, packed_bias)
        return self._packed[2], self._packed[3]

    def _fused_upsample(
        self,
        x_4d: torch.Tensor,
        batch_size: int,
        frames: int,
        shortcut: tuple[torch.Tensor, "DupUp3D", bool] | None,
    ) -> torch.Tensor:
        """2x2 conv on the low-res grid, then the phase gather (+ DupUp3D shortcut)."""
        packed, packed_bias = self._packed_weights()
        cout = packed.shape[0] // 4
        factor_t, repeats = (
            (1, 1) if shortcut is None else (shortcut[1].factor_t, shortcut[1].repeats)
        )
        if not can_use_cuda_upsample_gather(cout, factor_t, repeats):
            raise ValueError(
                f"the upsample gather kernel does not take {cout} output channels with a "
                f"temporal factor {factor_t} and {repeats} shortcut repeats"
            )
        defer_bias = packed_bias is not None
        if self.shard is not None:
            # One low-res line from each neighbour; the 2x2 conv then emits the H+1 (or W+1)
            # temp lines this rank's output rows read.
            temp = F.conv2d(
                self.shard.pad_frames(x_4d),
                packed,
                None if defer_bias else packed_bias,
                padding=self.shard.conv_padding((1, 1)),
            )
        else:
            temp = F.conv2d(x_4d, packed, None if defer_bias else packed_bias, padding=1)
        temp = _channels_last_2d_if_needed(temp)
        gather_bias = self.resample[1].bias if defer_bias else None
        if shortcut is None:
            return upsample_gather_cuda(temp, None, batch_size, frames, bias=gather_bias)
        residual, dup, first_chunk = shortcut
        residual = _channels_last_3d_if_needed(residual)
        crop = dup.factor_t - 1 if first_chunk else 0
        return upsample_gather_cuda(
            temp, residual, batch_size, frames, dup.factor_t, dup.repeats, crop, bias=gather_bias
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
        feat_idx: list[int] | None = None,
        shortcut: tuple[torch.Tensor, "DupUp3D", bool] | None = None,
    ) -> torch.Tensor:
        """Resample ``x`` ``[B, C, T, H, W]``: the upsample3d time step with its causal cache,
        then the spatial step (upsampling as a 2x2 conv on the low-res grid plus a phase
        gather), then (downsample3d) the temporal downsample continuation. ``shortcut`` =
        (residual, DupUp3D module, first_chunk): when given, the upsample adds the expanded
        shortcut while writing and the caller skips its add."""
        batch_size, channels, _, height, width = x.size()
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")
        if shortcut is not None and not self.mode.startswith("upsample"):
            raise ValueError("shortcut is only consumed by the upsample modes")

        if self.mode == "upsample3d" and feat_cache is not None and feat_idx is not None:
            x = self._temporal_upsample(x, feat_cache, feat_idx)

        frames = x.shape[2]
        x_4d = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x_4d = _channels_last_2d_if_needed(x_4d)
        if self.mode.startswith("upsample"):
            return self._fused_upsample(x_4d, batch_size, frames, shortcut)
        y = self._downsample(x_4d, batch_size)
        return self._after_resample(y, batch_size, frames, feat_cache, feat_idx)

    def _temporal_upsample(
        self, x: torch.Tensor, feat_cache: list[CacheEntry], feat_idx: list[int]
    ) -> torch.Tensor:
        """The upsample3d time step under the causal cache at ``feat_idx[0]`` (advanced by one).
        First chunk: records the ``"Rep"`` sentinel and returns ``x`` unchanged (its frames are
        repeated by the caller's layout). Later chunks: the time conv runs against the cached
        frame (zero history right after the sentinel), the chunk's last frames become the next
        cache entry, and the two channel halves are interleaved along time with the conv bias:
        ``[B, C, T, H, W] -> [B, C, 2T, H, W]``."""
        batch_size, channels, frames, height, width = x.size()
        idx = feat_idx[0]
        if feat_cache[idx] is None:
            feat_cache[idx] = "Rep"
            feat_idx[0] += 1
            return x

        cache_x = x[:, :, -CACHE_T:, :, :]
        if cache_x.shape[2] < 2 and feat_cache[idx] != "Rep":
            cache_x = torch.cat(
                [
                    _to_device_if_needed(
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x.device
                    ),
                    cache_x,
                ],
                dim=2,
            )
        if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
            cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
        fuse_interleave = cuda_frame_ops_available() and batch_size == 1
        self.time_conv.defer_bias = fuse_interleave
        if feat_cache[idx] == "Rep":
            x = self.time_conv(x)
        else:
            x = self.time_conv(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1

        if fuse_interleave:
            # Interleave the two channel halves along time and add the time conv's bias.
            u = frames_view(_channels_last_3d_if_needed(x))
            return frames_to_video(interleave_bias_cuda(u, self.time_conv.bias), height, width)
        x = x.reshape(batch_size, 2, channels, frames, height, width)
        x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
        return x.reshape(batch_size, channels, frames * 2, height, width)

    def _downsample(
        self, x_4d: torch.Tensor, batch_size: int, *, prepadded: bool = False
    ) -> torch.Tensor:
        """The stride-2 3x3 conv of the downsample modes on frames ``[B*T, C, H, W]``, after its
        zero pad (the shard's pad under sharding), or on an input that already carries the zero
        row and column (``prepadded``). When the owning residual down block consumes the bias
        (``defer_bias_to_consumer``), the conv runs bias-free and leaves it in ``pending_bias``."""
        if prepadded:
            x_4d = _channels_last_2d_if_needed(x_4d)
            if self.shard is not None:
                self.shard.fill_pad_line(x_4d)
        elif self.shard is None:
            x_4d = self.resample[0](x_4d)
        else:
            x_4d = self.shard.pad_for_downsample(_channels_last_2d_if_needed(x_4d))
        conv = self.resample[1]
        defer = (
            self.defer_bias_to_consumer
            and conv.bias is not None
            and cuda_frame_ops_available()
            and batch_size == 1
        )
        self.pending_bias = conv.bias if defer else None
        if defer:
            return F.conv2d(x_4d, conv.weight, None, conv.stride, conv.padding)
        return conv(x_4d)

    def take_pending_bias(self) -> torch.Tensor | None:
        bias, self.pending_bias = self.pending_bias, None
        return bias

    def _time_conv_bias(self, bias: torch.Tensor) -> torch.Tensor:
        """``time_conv``'s bias with the downsample conv's bias folded through its taps:
        every input frame of the stride-2 time conv carries ``bias``, so
        ``b' = b_t + sum_k W_t[:, :, k] @ bias`` (composed in fp32, rounded once)."""
        tc = self.time_conv
        key = (tc.weight.data_ptr(), tc.weight._version, bias.data_ptr(), bias._version)
        if self._time_bias is None or self._time_bias[:4] != key:
            with torch.inference_mode(False), torch.no_grad():
                w = tc.weight.detach().float()[:, :, :, 0, 0]  # [Cout, Cin, taps]
                b = bias.detach().float()
                bt = tc.bias.detach().float() if tc.bias is not None else torch.zeros_like(b)
                folded = (bt + torch.einsum("oik,i->o", w, b)).to(tc.weight.dtype)
            self._time_bias = (*key, folded)
        return self._time_bias[4]

    def forward_prepadded(
        self,
        x_4d: torch.Tensor,
        frames: int,
        feat_cache: list[CacheEntry] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        """Downsample path on an input that already carries the zero row/column the
        ``ZeroPad2d`` would add; ``x_4d`` is ``[T, C, H+1, W+1]`` channels-last, batch 1."""
        if not self.mode.startswith("downsample"):
            raise ValueError("forward_prepadded is for the downsample modes")
        x = self._downsample(x_4d, 1, prepadded=True)
        return self._after_resample(x, 1, frames, feat_cache, feat_idx)

    def _after_resample(
        self,
        x: torch.Tensor,
        batch_size: int,
        frames: int,
        feat_cache: list[CacheEntry] | None,
        feat_idx: list[int] | None,
    ) -> torch.Tensor:
        x_5d = x.reshape(batch_size, frames, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        x = _channels_last_3d_if_needed(x_5d)

        if self.mode == "downsample3d" and feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :]
                x_in = torch.cat([feat_cache[idx][:, :, -1:, :, :], x], dim=2)
                bias = self.take_pending_bias()
                if bias is not None:
                    tc = self.time_conv
                    x = F.conv3d(x_in, tc.weight, self._time_conv_bias(bias), tc.stride)
                else:
                    x = self.time_conv(x_in)
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


def _norm_into_buffer(
    norm: WinoVaeRMSNorm,
    conv: WinoVaeCausalConv3d,
    x: "torch.Tensor | PendingAdd",
    feat_cache: list[CacheEntry],
    feat_idx: list[int],
    spatial: tuple[int, int],
    return_sum: bool = False,
) -> torch.Tensor | None:
    """Start ``conv``'s next frame-buffer chunk (a logical ``spatial`` frame) and have ``norm``
    write ``SiLU(norm(x))`` into its window, halo lines completed; ``conv.buffer_conv`` then
    reads it. With ``return_sum``, returns ``x`` as a tensor (formed by the norm kernel)."""
    idx = feat_idx[0]
    window = conv.buffer_begin(feat_cache[idx], x.shape[2], x, spatial)
    feat_cache[idx] = conv
    feat_idx[0] += 1
    out = norm.norm_silu(
        x, out=window, out_geometry=conv.frame_buffer.geometry, return_sum=return_sum
    )
    conv.frame_buffer.exchange(window)
    return out[0] if return_sum else None


class WinoVaeResidualBlock(WanResidualBlock):
    """``WanResidualBlock`` on the fused paths: the frame-buffer convs, the temporal Winograd
    convs, or the eager convs with deferred biases."""

    _norm_cls = WinoVaeRMSNorm
    _conv_cls = WinoVaeCausalConv3d

    def __init__(
        self, in_dim: int, out_dim: int, dropout: float = 0.0, non_linearity: str = "silu"
    ) -> None:
        super().__init__(in_dim, out_dim, dropout, non_linearity)
        self._merged_bias: tuple[int, int, torch.Tensor] | None = None
        # Spatial Winograd applies to both convs on every path (frame pairs, cached single
        # frames), so it is fixed here, not when a path first runs.
        spatial = cuda_op_available() and cuda_wino_spatial_available()
        for conv in (self.conv1, self.conv2):
            conv.spatial_winograd = spatial
            conv.cudnn_frontend = True

    def _output_bias(self, with_shortcut: bool) -> torch.Tensor:
        """conv2's bias, plus the shortcut conv's when that one was deferred too."""
        if not with_shortcut:
            return self.conv2.bias
        b2, bs = self.conv2.bias, self.conv_shortcut.bias
        key = (b2._version, bs._version)
        if self._merged_bias is None or self._merged_bias[:2] != key:
            self._merged_bias = (*key, (b2.detach().float() + bs.detach().float()).to(b2.dtype))
        return self._merged_bias[2]

    def forward(
        self,
        x: "torch.Tensor | PendingAdd",
        feat_cache: list[CacheEntry] | None = None,
        feat_idx: list[int] | None = None,
        *,
        defer_output_bias: bool = False,
        return_input: bool = False,
    ) -> "PendingAdd | tuple[PendingAdd, torch.Tensor]":
        """The block on the fastest path its input allows: both convs as temporal Winograd
        (an even number of frames), as frame-buffer convs (the norm kernels write into the
        buffers), or eagerly. Returns ``conv2(...) + shortcut`` as a PendingAdd the next norm
        adds while reading; ``defer_output_bias``: the consumer is a fused norm, so conv2's
        bias (and a shortcut conv's) ride on it too. ``return_input``: also return the
        materialized block input."""
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")
        batch, _, frames = x.shape[:3]
        cached = feat_cache is not None and cuda_op_available()
        if (
            cached
            and self.conv1.winograd_eligible(frames, batch)
            and self.conv2.winograd_eligible(frames, batch)
        ):
            out, x = self._forward_winograd(x, feat_cache, feat_idx, frames)
        elif (
            cached
            and self.conv1.buffer_eligible(x)
            and self.conv2.buffer_eligible(x)
            and not (frames == 1 and feat_cache[feat_idx[0]] is None)
        ):
            out, x = self._forward_buffered(x, feat_cache, feat_idx, defer_output_bias)
        else:
            out, x = self._forward_eager(x, feat_cache, feat_idx, defer_output_bias)
        return (out, x) if return_input else out

    def _shortcut(
        self, x: torch.Tensor, defer_out: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The shortcut branch and the bias the block output carries: with ``defer_out``,
        conv2's bias plus a shortcut conv's (whose own bias is then left out)."""
        has_conv = isinstance(self.conv_shortcut, WinoVaeCausalConv3d)
        if has_conv:
            self.conv_shortcut.defer_bias = defer_out
        residual = self.conv_shortcut(x)
        return residual, (self._output_bias(has_conv) if defer_out else None)

    def _forward_eager(
        self,
        x: "torch.Tensor | PendingAdd",
        feat_cache: list[CacheEntry] | None,
        feat_idx: list[int] | None,
        defer_out: bool,
    ) -> tuple[PendingAdd, torch.Tensor]:
        """The convs through torch (the history prepended by copy); biases still deferred."""
        if feat_cache is not None:
            for i in (feat_idx[0], feat_idx[0] + 1):
                feat_cache[i] = _cache_frames(feat_cache[i])
        x, h = self.norm1.norm_silu(x, return_sum=True)
        residual, out_bias = self._shortcut(x, defer_out)
        self.conv1.defer_bias = True
        if feat_cache is not None:
            h = _causal_conv_with_cache(self.conv1, h, feat_cache, feat_idx)
        else:
            h = self.conv1(h)
        h = self.norm2.norm_silu(PendingAdd(h, None, self.conv1.bias))
        h = self.dropout(h)
        self.conv2.defer_bias = defer_out
        if feat_cache is not None:
            h = _causal_conv_with_cache(self.conv2, h, feat_cache, feat_idx)
        else:
            h = self.conv2(h)
        return PendingAdd(h, residual, out_bias), x

    def _forward_buffered(
        self,
        x: "torch.Tensor | PendingAdd",
        feat_cache: list[CacheEntry],
        feat_idx: list[int],
        defer_out: bool,
    ) -> tuple[PendingAdd, torch.Tensor]:
        """Plain 3D convs reading their frame buffers; the norm kernels write into them."""
        frames = x.shape[2]
        spatial = tuple(x.shape[-2:])
        x = _norm_into_buffer(self.norm1, self.conv1, x, feat_cache, feat_idx, spatial, True)
        residual, out_bias = self._shortcut(x, defer_out)
        self.conv1.defer_bias = True
        h = self.conv1.buffer_conv(frames)
        _norm_into_buffer(
            self.norm2,
            self.conv2,
            PendingAdd(h, None, self.conv1.bias),
            feat_cache,
            feat_idx,
            spatial,
        )
        self.conv2.defer_bias = defer_out
        h = self.conv2.buffer_conv(frames)
        return PendingAdd(h, residual, out_bias), x

    def _forward_winograd(
        self,
        x: "torch.Tensor | PendingAdd",
        feat_cache: list[CacheEntry],
        feat_idx: list[int],
        frames: int,
    ) -> tuple[PendingAdd, torch.Tensor]:
        """Both convs as temporal Winograd. The caches hold the last two normalized frames
        exactly as the eager path stores them (a frame-buffer conv left by an odd-length
        chunk hands over its history frames); conv biases ride on the PendingAdd and are added
        by the consumer kernel (free there)."""
        h, w = x.shape[-2:]
        order = temporal_order_for(frames)
        # norm1 -> inputs for conv1; the sum it writes is the block input (residual source).
        s, inputs = self._norm_winograd(self.norm1, self.conv1, x, feat_cache, feat_idx, order)
        if isinstance(x, PendingAdd):
            x = s if s is not None else x.materialize()
        residual, out_bias = self._shortcut(x, True)
        m1, geo_m1 = self.conv1.forward_winograd(inputs)
        products = PendingAdd(None, None, self.conv1.bias, wino=m1, geometry=geo_m1)
        _, inputs2 = self._norm_winograd(
            self.norm2, self.conv2, products, feat_cache, feat_idx, order
        )
        m2, geo_m2 = self.conv2.forward_winograd(inputs2)
        return PendingAdd(None, residual, out_bias, wino=m2, geometry=geo_m2), x

    @staticmethod
    def _norm_winograd(
        norm: WinoVaeRMSNorm,
        conv: WinoVaeCausalConv3d,
        x: "torch.Tensor | PendingAdd",
        feat_cache: list[CacheEntry],
        feat_idx: list[int],
        order: int,
    ) -> tuple[torch.Tensor | None, WinogradInputs]:
        """``norm`` writes the Winograd inputs of ``conv`` (halo lines completed) and the next
        cache entry; returns the sum it formed (None for a plain tensor input) and the inputs."""
        h, w = x.shape[-2:]
        frames = x.shape[2]
        idx = feat_idx[0]
        geo, into, direct = conv.winograd_inputs(h, w, order, frames)
        s, cache, inputs = norm.norm_silu_winograd(
            x, _cache_frames(feat_cache[idx]), frames, geo, inputs=into
        )
        feat_cache[idx] = cache
        feat_idx[0] += 1
        if direct is not None:
            direct.barrier()
        elif conv.shard is not None:
            conv.shard.zero_outside(inputs.physical(), inputs.geometry)
            conv.shard.exchange(inputs.physical(), inputs.geometry)
        return s, inputs


class WinoVaeAttentionBlock(WanAttentionBlock):
    """The mid-block attention with the norm, projections and residual add fused."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.shard: Shard | None = None

    def _forward_gathered(self, x: torch.Tensor) -> torch.Tensor:
        """The eager attention on the whole frame: gather the ranks' slices, run, keep ours."""
        shard = self.shard
        dim = 3 if shard.axis == "h" else 4
        extents = shard.output_extents(x.shape[dim])
        widest = max(extents)
        send = x.contiguous()
        if x.shape[dim] != widest:
            shape = list(x.shape)
            shape[dim] = widest
            send = x.new_zeros(shape)
            send.narrow(dim, 0, x.shape[dim]).copy_(x)
        parts = [torch.empty_like(send) for _ in range(shard.world_size)]
        torch.distributed.all_gather(parts, send, group=shard.pg)
        full = super().forward(
            torch.cat([p.narrow(dim, 0, e) for p, e in zip(parts, extents)], dim=dim)
        )
        return torch.narrow(full, dim, sum(extents[: shard.rank]), x.shape[dim])

    def forward(self, x: "torch.Tensor | PendingAdd") -> "torch.Tensor | PendingAdd":  # type: ignore[override]
        x = _as_tensor(x)
        if not (cuda_op_available() and x.shape[0] == 1 and can_use_cuda_rms_norm_silu(x)):
            return super().forward(x) if self.shard is None else self._forward_gathered(x)
        residual = _channels_last_3d_if_needed(x)
        _, channels, frames, height, width = residual.shape
        positions = height * width
        # The GEMMs run on the channels-last row view [T*H*W, C], which is also SDPA's layout.
        normed = rms_norm_silu_cuda(
            residual,
            self.norm.gamma,
            None,
            self.norm.scale,
            silu=False,
        )
        normed_rows = frames_view(normed).reshape(frames * positions, channels)
        w_qkv = self.to_qkv.weight.reshape(3 * channels, channels)
        if self.shard is None:
            qkv = torch.addmm(self.to_qkv.bias, normed_rows, w_qkv.t())  # bias in the GEMM epilogue
            qkv = qkv.view(frames, 1, positions, 3 * channels)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            # Attention spans the whole frame: keys and values from every rank's rows (their
            # order does not matter), queries from ours.
            q = torch.addmm(self.to_qkv.bias[:channels], normed_rows, w_qkv[:channels].t()).view(
                frames, 1, positions, channels
            )
            all_rows = self.shard.gather_rows(normed_rows)
            kv = torch.addmm(self.to_qkv.bias[channels:], all_rows, w_qkv[channels:].t())
            if self.shard.uneven:
                # Rank-major rows of unequal counts: per rank [T, P_r, 2C], joined along P.
                counts = self.shard.row_counts(normed_rows.shape[0])
                per_rank = [
                    part.view(frames, n // frames, 2 * channels)
                    for part, n in zip(kv.split(counts), counts)
                ]
                kv = torch.cat(per_rank, dim=1).unsqueeze(1)
            else:
                kv = kv.view(self.shard.world_size, frames, positions, 2 * channels)
                kv = kv.permute(1, 0, 2, 3).reshape(frames, 1, -1, 2 * channels)
            k, v = kv.chunk(2, dim=-1)
        attn = F.scaled_dot_product_attention(q, k, v)  # [T, 1, HW, C]
        attn_rows = attn.reshape(frames * positions, channels)
        w_proj = self.proj.weight.reshape(channels, channels)
        proj_rows = torch.mm(attn_rows, w_proj.t())
        proj = frames_to_video(proj_rows.view(frames, positions, channels), height, width)
        # x + residual (+ the proj bias) is added by the consumer's norm kernel.
        return PendingAdd(proj, residual, self.proj.bias)


def _run_resnets(
    resnets: nn.ModuleList,
    x: "torch.Tensor | PendingAdd",
    feat_cache: list[CacheEntry] | None,
    feat_idx: list[int] | None,
    *,
    defer_last_bias: bool,
    defer_first_bias: bool = True,
) -> tuple["torch.Tensor | PendingAdd", torch.Tensor]:
    """Run a chain of residual blocks; returns ``(output, materialized input)``.

    The first block consumes a PendingAdd input inside its norm kernel and hands back
    the sum, so the caller gets the block input without a separate add. Every block
    but the last defers its conv2 bias to the next block's norm; the last one does so
    only when ``defer_last_bias`` (its consumer is a fused norm)."""
    last = len(resnets) - 1
    entry: torch.Tensor | None = None
    for i, resnet in enumerate(resnets):
        defer = (defer_last_bias if i == last else True) and (defer_first_bias or i > 0)
        if i == 0:
            x, entry = resnet(
                x,
                feat_cache=feat_cache,
                feat_idx=feat_idx,
                defer_output_bias=defer,
                return_input=True,
            )
        else:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx, defer_output_bias=defer)
    assert entry is not None
    return x, entry


class WinoVaeMidBlock(WanMidBlock):
    _resblock_cls = WinoVaeResidualBlock
    _attention_cls = WinoVaeAttentionBlock

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        # The fused attention block returns its residual add as a PendingAdd for the next
        # norm; the plain one needs a materialized tensor.
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx, defer_output_bias=False)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx, defer_output_bias=True)
        return x


class WinoVaeResidualDownBlock(WanResidualDownBlock):
    _resblock_cls = WinoVaeResidualBlock
    _resample_cls = WinoVaeResample

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.downsampler is not None:
            self.downsampler.defer_bias_to_consumer = True

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        # The last block feeds the downsampler or the shortcut add, never a norm.
        x, residual = _run_resnets(self.resnets, x, feat_cache, feat_idx, defer_last_bias=False)
        if self.downsampler is not None:
            if isinstance(x, PendingAdd) and x.can_fuse():
                # One pass writes the downsampler's zero-padded input directly.
                x = self.downsampler.forward_prepadded(
                    x.materialize_padded_2d(), x.frames, feat_cache=feat_cache, feat_idx=feat_idx
                )
            else:
                x = self.downsampler(_as_tensor(x), feat_cache=feat_cache, feat_idx=feat_idx)
        else:
            x = _as_tensor(x)
        x = _channels_last_3d_if_needed(x)
        # The downsample conv's bias, when it ran without one (downsample2d; the time conv
        # absorbs it in downsample3d).
        bias = self.downsampler.take_pending_bias() if self.downsampler is not None else None
        dup = self.avg_shortcut
        if cuda_frame_ops_available() and x.shape[0] == 1:
            residual = _channels_last_3d_if_needed(residual)
            t_in = residual.shape[2]
            pad_t = (dup.factor_t - t_in % dup.factor_t) % dup.factor_t
            h_out, w_out = x.shape[-2:]
            y = avg_down_add_cuda(
                frames_view(x),
                frames_view(residual),
                h_out,
                w_out,
                residual.shape[-1],
                dup.factor_t,
                dup.factor_s,
                dup.group_size,
                pad_t,
                bias,
            )
            return frames_to_video(y, h_out, w_out)
        if bias is not None:
            x = x + bias.view(1, -1, 1, 1, 1)
        return _channels_last_3d_if_needed(x + dup(residual))


def _conv_in(
    conv: WinoVaeCausalConv3d,
    x: torch.Tensor,
    feat_cache: list[CacheEntry] | None,
    feat_idx: list[int] | None,
) -> PendingAdd:
    """The model's first conv, its bias left for the next norm kernel to add while reading."""
    conv.defer_bias = True
    if feat_cache is not None:
        x = _causal_conv_with_cache(conv, x, feat_cache, feat_idx)
    else:
        x = conv(x)
    return PendingAdd(x, None, conv.bias)


def _norm_silu_conv(
    norm: WinoVaeRMSNorm,
    conv: WinoVaeCausalConv3d,
    x: "torch.Tensor | PendingAdd",
    feat_cache: list[CacheEntry] | None,
    feat_idx: list[int] | None,
) -> torch.Tensor:
    """``conv(SiLU(norm(x)))`` with the causal cache: the norm kernel writes straight into the
    conv's frame buffer where it can, else the eager norm and conv."""
    frames = x.shape[2]
    if (
        cuda_op_available()
        and feat_cache is not None
        and conv.buffer_eligible(x)
        and not (frames == 1 and feat_cache[feat_idx[0]] is None)
    ):
        _norm_into_buffer(norm, conv, x, feat_cache, feat_idx, tuple(x.shape[-2:]))
        return conv.buffer_conv(frames)
    x = norm.norm_silu(x)
    if feat_cache is not None:
        x = _causal_conv_with_cache(conv, x, feat_cache, feat_idx)
    else:
        x = conv(x)
    return _channels_last_3d_if_needed(x)


class WinoVaeEncoder3d(WanEncoder3d):
    _conv_cls = WinoVaeCausalConv3d
    _down_block_cls = WinoVaeResidualDownBlock
    _resblock_cls = WinoVaeResidualBlock
    _resample_cls = WinoVaeResample
    _mid_block_cls = WinoVaeMidBlock
    _norm_cls = WinoVaeRMSNorm

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Temporary folded-tail submodule for one encode; assigned by WinoVae._encode_impl and
        # cleared in its finally. It contributes no state-dict entries after that encode returns.
        self.conv_out_override: WinoVaeCausalConv3d | None = None

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")

        x = _conv_in(self.conv_in, x, feat_cache, feat_idx)

        for layer in self.down_blocks:
            if not isinstance(layer, (WinoVaeResidualBlock, WinoVaeResidualDownBlock)):
                x = _as_tensor(x)
            if feat_cache is not None:
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        conv_out = self.conv_out_override or self.conv_out
        return _norm_silu_conv(self.norm_out, conv_out, x, feat_cache, feat_idx)


class WinoVaeResidualUpBlock(WanResidualUpBlock):
    _resblock_cls = WinoVaeResidualBlock
    _resample_cls = WinoVaeResample

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        # With an upsampler the last block feeds a conv, so its bias stays in eager; the
        # final up block (no upsampler) hands its output to norm_out.
        x, residual = _run_resnets(
            self.resnets, x, feat_cache, feat_idx, defer_last_bias=self.upsampler is None
        )
        if self.upsampler is not None:
            fuse_shortcut = self.avg_shortcut is not None
            x = self.upsampler(
                _as_tensor(x),
                feat_cache=feat_cache,
                feat_idx=feat_idx,
                shortcut=(residual, self.avg_shortcut, first_chunk) if fuse_shortcut else None,
            )
            if fuse_shortcut:
                return x
        if self.avg_shortcut is not None:
            x = _as_tensor(x) + self.avg_shortcut(residual, first_chunk=first_chunk)
        return x if isinstance(x, PendingAdd) else _channels_last_3d_if_needed(x)


class WinoVaeUpBlock(WanUpBlock):
    _resblock_cls = WinoVaeResidualBlock
    _resample_cls = WinoVaeResample

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool | None = None,
    ) -> torch.Tensor:
        del first_chunk
        x, _ = _run_resnets(
            self.resnets, x, feat_cache, feat_idx, defer_last_bias=self.upsamplers is None
        )
        if self.upsamplers is not None:
            x = self.upsamplers[0](_as_tensor(x), feat_cache=feat_cache, feat_idx=feat_idx)
        return x if isinstance(x, PendingAdd) else _channels_last_3d_if_needed(x)


class WinoVaeDecoder3d(WanDecoder3d):
    _conv_cls = WinoVaeCausalConv3d
    _mid_block_cls = WinoVaeMidBlock
    _res_up_block_cls = WinoVaeResidualUpBlock
    _up_block_cls = WinoVaeUpBlock
    _norm_cls = WinoVaeRMSNorm

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[CacheEntry] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")

        x = _conv_in(self.conv_in, x, feat_cache, feat_idx)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

        return _norm_silu_conv(self.norm_out, self.conv_out, x, feat_cache, feat_idx)


class LazyDiagonalGaussianDistribution(DiagonalGaussianDistribution):
    """``DiagonalGaussianDistribution`` whose ``std``/``var``/clamped ``logvar`` are computed
    on first use: ``mode()`` launches nothing beyond the view split."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self._logvar_raw = torch.chunk(parameters, 2, dim=1)
        self.deterministic = deterministic
        self._logvar: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._var: torch.Tensor | None = None

    @property
    def logvar(self) -> torch.Tensor:
        if self._logvar is None:
            self._logvar = torch.clamp(self._logvar_raw, -30.0, 20.0)
        return self._logvar

    @property
    def std(self) -> torch.Tensor:
        if self._std is None:
            self._std = (
                torch.zeros_like(self.mean) if self.deterministic else torch.exp(0.5 * self.logvar)
            )
        return self._std

    @property
    def var(self) -> torch.Tensor:
        if self._var is None:
            self._var = (
                torch.zeros_like(self.mean) if self.deterministic else torch.exp(self.logvar)
            )
        return self._var


class WinoVae(WanVAE):
    """Optimized Wan VAE; state-dict compatible with ``WanVAE`` and ``AutoencoderKLWan``.

    The model and its inputs must be bf16 on CUDA (the fused kernels read and write bf16;
    ``encode`` / ``decode`` raise ``TypeError`` otherwise).

    Beyond the ``AutoencoderKLWan`` interface: ``encode_mode(x)`` returns the posterior mean
    directly, and ``fold_latent_affine(mean, std)`` folds the
    latent de-normalization into ``post_quant_conv`` so ``decode`` takes the model's raw
    latents.
    """

    _encoder_cls = WinoVaeEncoder3d
    _decoder_cls = WinoVaeDecoder3d
    _conv_cls = WinoVaeCausalConv3d
    # Spatial sharding (set_shard); a class default so the base constructor's clear_cache sees it.
    shard: Shard | None = None

    def __init__(
        self,
        config: WanVAEConfig | dict[str, Any] | None = None,
        options: WinoVaeOptions | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        if options is None:
            options = WinoVaeOptions()
        if isinstance(options, dict):
            options = WinoVaeOptions.from_dict(options)
        self.options = options
        time_stages = sum(bool(s) for s in self.config.temperal_downsample)
        if (
            self.config.scale_factor_temporal is not None
            and self.config.scale_factor_temporal != 2**time_stages
        ):
            raise ValueError(
                f"scale_factor_temporal={self.config.scale_factor_temporal} does not match "
                f"temperal_downsample={self.config.temperal_downsample}, which compresses time "
                f"by {2**time_stages}."
            )
        self.decoder.conv_out.max_in_frames = _CONV_OUT_MAX_IN_FRAMES
        for module in self.modules():
            if isinstance(module, WinoVaeCausalConv3d):
                module.frame_buffer.chunks = options.frame_buffer_chunks
        self._folded_tail: dict[int | None, tuple[tuple, WinoVaeCausalConv3d]] = {}
        self._post_quant_folded: tuple[torch.Tensor, torch.Tensor] | None = None

    def _benchmark_convs(self, *, frames: int, encoder: bool) -> bool:
        """cuDNN autotuning for the encoder and image decodes, except on a spatial shard, where the heuristics'
        choice is the same in every process (autotuning could pick different engines per rank)."""
        return self.shard is None and _benchmark_convs(encoder=encoder, frames=frames)

    def set_shard(self, shard: Shard | None) -> None:
        """Run on one spatial slice of the frames: every module exchanges the halo lines it
        needs with the neighbouring ranks (see :mod:`wino_vae_shard`). The caller splits the
        latents / pixels along the shard's axis and gathers the outputs."""
        self.shard = shard
        for module in self.modules():
            if isinstance(
                module,
                (WinoVaeCausalConv3d, WinoVaeRMSNorm, WinoVaeResample, WinoVaeAttentionBlock),
            ):
                module.shard = shard
        self._folded_tail.clear()

    def clear_cache(self) -> None:
        if self.shard is not None:
            self.shard.begin_call()
        super().clear_cache()

    def _check_dtype(self, x: torch.Tensor) -> None:
        """The kernels read and write bf16 CUDA tensors."""
        if not x.is_cuda or x.dtype is not torch.bfloat16 or self.dtype is not torch.bfloat16:
            raise TypeError(
                "WinoVae needs bf16 CUDA inputs and weights; got input "
                f"{x.dtype} on {x.device} and weights {self.dtype}"
            )

    def _encode(self, x: torch.Tensor, *, mode_only: bool = False) -> torch.Tensor:
        self._check_dtype(x)
        prev_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._benchmark_convs(encoder=True, frames=x.shape[2])
        try:
            return self._encode_impl(x, mode_only=mode_only)
        finally:
            torch.backends.cudnn.benchmark = prev_benchmark

    def _folded_tail_conv(self, rows: int | None) -> WinoVaeCausalConv3d:
        """conv_out with quant_conv folded in: ``W' = Q A``, ``b' = Q a + q`` (composed in
        fp32, rounded once to the model dtype); ``rows`` keeps only the first output
        channels (the posterior mean) or all of them."""
        conv_out, quant = self.encoder.conv_out, self.quant_conv
        key = (
            conv_out.weight.data_ptr(),
            conv_out.weight._version,
            quant.weight.data_ptr(),
            quant.weight._version,
            rows,
        )
        cached = self._folded_tail.get(rows)
        if cached is not None and cached[0] == key:
            return cached[1]
        # Built outside inference mode even when called from it: the folded conv's weight
        # caches key on the tensor version counter, which inference tensors do not carry.
        with torch.inference_mode(False), torch.no_grad():
            folded = self._build_folded_tail(conv_out, quant, rows)
        self._folded_tail[rows] = (key, folded)
        return folded

    @staticmethod
    def _build_folded_tail(
        conv_out: WinoVaeCausalConv3d, quant: WinoVaeCausalConv3d, rows: int | None
    ) -> WinoVaeCausalConv3d:
        q = quant.weight.detach().float().flatten(1)  # [2z, 2z]
        qb = quant.bias.detach().float() if quant.bias is not None else torch.zeros_like(q[:, 0])
        a = conv_out.weight.detach().float()  # [2z, Cin, 3, 3, 3]
        ab = (
            conv_out.bias.detach().float()
            if conv_out.bias is not None
            else torch.zeros_like(q[:, 0])
        )
        if rows is not None:
            q, qb = q[:rows], qb[:rows]
        w = torch.einsum("om,mikhw->oikhw", q, a)
        b = q @ ab + qb
        folded = WinoVaeCausalConv3d(
            conv_out.in_channels, w.shape[0], conv_out.kernel_size, padding=conv_out.padding[1]
        ).to(device=a.device, dtype=conv_out.weight.dtype)
        folded.weight.copy_(w)
        folded.bias.copy_(b)
        folded.weight.requires_grad_(False)
        folded.bias.requires_grad_(False)
        folded.max_in_frames = conv_out.max_in_frames
        folded.frame_buffer.chunks = conv_out.frame_buffer.chunks
        folded.shard = conv_out.shard
        return folded

    def _video_for(self, chunk: torch.Tensor, latent_frames: int) -> torch.Tensor:
        """The uninitialized channels-last output video ``[1, C, T, H, W]`` of a decode of
        ``latent_frames`` latents, shaped from its first decoded (patchified) chunk."""
        patch = self.config.patch_size
        frames = (latent_frames - 1) * self.config.scale_factor_temporal + 1
        return torch.empty(
            (
                1,
                chunk.shape[1] // (patch * patch),
                frames,
                chunk.shape[3] * patch,
                chunk.shape[4] * patch,
            ),
            dtype=chunk.dtype,
            device=chunk.device,
            memory_format=torch.channels_last_3d,
        )

    def _encode_impl(self, x: torch.Tensor, *, mode_only: bool = False) -> torch.Tensor:
        _, _, num_frame, _, _ = x.shape

        self.clear_cache()
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)
        x = _channels_last_3d_if_needed(x)

        self.encoder.conv_out_override = self._folded_tail_conv(self.z_dim if mode_only else None)
        # The first pass is one frame; every later pass is a multiple of four frames so the
        # stride-2 temporal downsamplers see chunk boundaries at the same parity each time.
        step = self.options.encoder_chunk_frames
        bounds = [(0, 1)] + [
            (start, min(start + step, num_frame)) for start in range(1, num_frame, step)
        ]
        out_chunks: list[torch.Tensor] = []
        try:
            for start, end in bounds:
                self._enc_conv_idx = [0]
                out_chunks.append(
                    self.encoder(
                        x[:, :, start:end, :, :],
                        feat_cache=self._enc_feat_map,
                        feat_idx=self._enc_conv_idx,
                    )
                )
        finally:
            self.encoder.conv_out_override = None

        if len(out_chunks) == 1:
            out = out_chunks[0]
        else:
            out = torch.cat(out_chunks, dim=2)

        enc = _channels_last_3d_if_needed(out)
        self.clear_cache()
        return enc

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        h = self._encode(x)
        posterior = LazyDiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def encode_mode(self, x: torch.Tensor) -> torch.Tensor:
        """The posterior mean ``[B, z, T', H', W']`` of ``x`` without computing the
        log-variance half (the same values as ``encode(x).latent_dist.mode()``)."""
        return self._encode(x, mode_only=True)

    def _decode(
        self,
        z: torch.Tensor,
        return_dict: bool = True,
        *,
        temporal_chunk_size: int = 1,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        self._check_dtype(z)
        _, _, num_frame, _, _ = z.shape

        self.clear_cache()
        prev_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._benchmark_convs(encoder=False, frames=num_frame)
        try:
            return self._decode_impl(z, return_dict, temporal_chunk_size)
        finally:
            torch.backends.cudnn.benchmark = prev_benchmark

    def fold_latent_affine(self, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> None:
        """Fold the latent de-normalization into ``decode``: ``latents_mean`` and
        ``latents_std`` are per-channel (``z_dim`` elements each); afterwards ``decode(z)``
        behaves as ``decode(std * z + mean)`` until ``clear_latent_affine``. The registered
        weights are unchanged (the folded tensors are plain attributes, not buffers), so call
        this after loading, moving or casting the weights, and again after any such change."""
        # P' = P diag(std), b' = P mean + b, composed in fp32 and rounded once to the weight dtype.
        conv = self.post_quant_conv
        with torch.inference_mode(False), torch.no_grad():
            w = conv.weight.detach().float()  # [z, z, 1, 1, 1]
            z = w.shape[0]
            mean = latents_mean.detach().to(w.device, torch.float32).reshape(z)
            std = latents_std.detach().to(w.device, torch.float32).reshape(z)
            b = (
                conv.bias.detach().float()
                if conv.bias is not None
                else torch.zeros(z, device=w.device)
            )
            folded_w = (w * std.view(1, z, 1, 1, 1)).to(conv.weight.dtype)
            folded_b = (w.view(z, z) @ mean + b).to(conv.weight.dtype)
        self._post_quant_folded = (folded_w, folded_b)

    def clear_latent_affine(self) -> None:
        self._post_quant_folded = None

    def _post_quant(self, z: torch.Tensor) -> torch.Tensor:
        if self._post_quant_folded is None:
            return self.post_quant_conv(z)
        w, b = self._post_quant_folded
        return _channels_last_3d_if_needed(F.conv3d(z, w, b))

    def _use_fused_output(self, x: torch.Tensor) -> bool:
        return (
            self.config.patch_size is not None and x.shape[0] == 1 and unpatchify_clamp_available()
        )

    def _decode_impl(
        self, z: torch.Tensor, return_dict: bool, temporal_chunk_size: int
    ) -> DecoderOutput | tuple[torch.Tensor]:
        _, _, num_frame, _, _ = z.shape
        z = _channels_last_3d_if_needed(z)
        x = self._post_quant(z)
        fused = self._use_fused_output(x)
        out: torch.Tensor | None = None
        frames_done = 0
        out_chunks: list[torch.Tensor] = []
        for index, frame_slice in enumerate(_decode_chunk_slices(num_frame, temporal_chunk_size)):
            self._conv_idx = [0]
            out_chunk = self.decoder(
                x[:, :, frame_slice, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=index == 0,
            )
            if fused:
                # Unpatchify + clamp of this chunk straight into the preallocated video.
                out_chunk = _channels_last_3d_if_needed(out_chunk)
                if out is None:
                    out = self._video_for(out_chunk, num_frame)
                unpatchify_clamp_into(out_chunk, out, frames_done, self.config.patch_size)
                frames_done += out_chunk.shape[2]
            else:
                out_chunks.append(out_chunk)

        if fused:
            assert out is not None
            if frames_done != out.shape[2]:
                raise RuntimeError(
                    f"decoder produced {frames_done} frames for a {out.shape[2]}-frame output"
                )
        else:
            out = out_chunks[0] if len(out_chunks) == 1 else torch.cat(out_chunks, dim=2)
            if self.config.patch_size is not None:
                out = unpatchify(out, patch_size=self.config.patch_size)
            out = _channels_last_3d_if_needed(out)
            out = torch.clamp(out, min=-1.0, max=1.0)

        self.clear_cache()
        if not return_dict:
            return (out,)
        return DecoderOutput(sample=out)
