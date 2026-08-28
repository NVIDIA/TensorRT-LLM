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
"""Video-reference decoding on NVDEC (PyNvVideoCodec).

Encoded bytes are demuxed from memory, decoded on NVDEC, and only the
requested window is retained, resized to the caller's target resolution — so
retained memory is bounded by that target and at any instant only one
source-resolution frame is alive.

Counterpart to :mod:`tensorrt_llm.media.encoding`.

PyNvVideoCodec is imported function-locally: ``import tensorrt_llm`` on a
CPU-only host must never load the driver-linked extension.
"""

import functools
import math
from typing import NamedTuple

import torch


@functools.lru_cache(maxsize=32)
def _lanczos_taps(
    in_size: int, out_size: int, device_str: str, a: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local-support Lanczos-a taps: ``([out, K] weights, [out, K] indices)``.

    PIL semantics (the reference vllm-omni preprocess resizes with
    ``PIL.Image.Resampling.LANCZOS``): the kernel is stretched by the
    downscale ratio, taps span ``a * scale`` source pixels around each output
    center, out-of-range taps get zero weight, and each row normalizes.
    float32, like PIL's internal filter precision (PIL then quantizes
    coefficients — parity is bounded, not bit-exact).

    ``K = ceil(2 * a * max(in/out, 1)) + 1`` is the filter's true support, so
    applying these by gather + weighted sum costs ``O(out * K)`` per row
    instead of the ``O(out * in)`` of a dense resampling matrix. Cached per
    (in, out, device): every frame of a clip shares sizes, and the cached
    tensors are kilobytes.
    """
    device = torch.device(device_str)
    ratio = in_size / out_size
    scale = max(ratio, 1.0)
    support = a * scale
    centers = (torch.arange(out_size, device=device, dtype=torch.float32) + 0.5) * ratio
    first = torch.floor(centers - support)
    num_taps = int(math.ceil(2 * support)) + 1
    taps = first.unsqueeze(1) + torch.arange(num_taps, device=device, dtype=torch.float32)
    x = (taps + 0.5 - centers.unsqueeze(1)) / scale
    weights = torch.sinc(x) * torch.sinc(x / a)
    weights = torch.where(x.abs() < a, weights, torch.zeros_like(weights))
    valid = (taps >= 0) & (taps < in_size)
    weights = weights * valid
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights, taps.clamp(0, in_size - 1).long()


def _resample_last_dim(x: torch.Tensor, weights: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """Resample the last dim ``[..., in] -> [..., out]`` by gather + weighted sum."""
    return (x[..., taps] * weights).sum(-1)


def resize_center_crop_uint8(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize + center-crop uint8 ``[T, H, W, C]`` frames to the target size.

    Applied to the decoded reference frames before retention.
    Semantics mirror the reference implementation's PIL path (cover-scale by
    ``max(target/source)``, ceil-rounded resize with Lanczos-3, center crop),
    implemented as separable local-tap resampling: per output pixel only the
    filter's ``K`` support taps are gathered and summed.
    """
    t, h, w, c = frames.shape
    if (h, w) == (target_h, target_w):
        return frames
    ratio = max(target_w / w, target_h / h)
    resize_w = int(math.ceil(ratio * w))
    resize_h = int(math.ceil(ratio * h))

    # PIL resamples in two passes (horizontal, then vertical) and stores the
    # intermediate as uint8 — clamping away Lanczos overshoot between passes.
    # Same order and intermediate quantization here, so parity with the PIL
    # reference path stays within coefficient-rounding noise.
    x = frames.permute(0, 3, 1, 2).to(torch.float32)  # [T, C, H, W]
    if resize_w != w:
        weights, taps = _lanczos_taps(w, resize_w, str(frames.device))
        x = _resample_last_dim(x, weights, taps)
        x = x.round_().clamp_(0, 255)
    if resize_h != h:
        weights, taps = _lanczos_taps(h, resize_h, str(frames.device))
        x = _resample_last_dim(x.transpose(-1, -2), weights, taps).transpose(-1, -2)

    left = max((resize_w - target_w) // 2, 0)
    top = max((resize_h - target_h) // 2, 0)
    x = x[:, :, top : top + target_h, left : left + target_w]
    return x.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def resize_fit_pad_uint8(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize to fit inside the target, then pad bottom/right to fill it.

    The counterpart to :func:`resize_center_crop_uint8`, for references whose
    periphery carries signal — a robot gripper works at the frame edge, so
    cropping it away costs the model the thing it is meant to act on.

    Semantics mirror the action reference's ``reflection_pad_to_target``:
    contain-scale by ``min(target/source, 1.0)`` (never enlarge — a small clip
    keeps its own pixels and gets a wider border), round-rather-than-ceil
    resize, then pad bottom/right by reflection, switching to edge replication
    when a pad run reaches the resized extent (reflection has no source pixels
    left to mirror). The resampling filter stays this module's Lanczos-3 rather
    than the reference's bicubic: the geometry is what preserves content, and a
    second filter would buy sub-pixel differences for a second code path.
    """
    t, h, w, c = frames.shape
    if (h, w) == (target_h, target_w):
        return frames
    ratio = min(target_w / w, target_h / h, 1.0)
    resize_w = min(int(ratio * w + 0.5), target_w)
    resize_h = min(int(ratio * h + 0.5), target_h)

    # Two passes with a uint8-quantized intermediate, as in the cover path.
    x = frames.permute(0, 3, 1, 2).to(torch.float32)  # [T, C, H, W]
    if resize_w != w:
        weights, taps = _lanczos_taps(w, resize_w, str(frames.device))
        x = _resample_last_dim(x, weights, taps)
        x = x.round_().clamp_(0, 255)
    if resize_h != h:
        weights, taps = _lanczos_taps(h, resize_h, str(frames.device))
        x = _resample_last_dim(x.transpose(-1, -2), weights, taps).transpose(-1, -2)
    x = x.round_().clamp_(0, 255)

    pad_w = target_w - resize_w
    pad_h = target_h - resize_h
    if pad_w or pad_h:
        mode = "replicate" if (pad_w >= resize_w or pad_h >= resize_h) else "reflect"
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x.to(torch.uint8).permute(0, 2, 3, 1).contiguous()


_RESIZE_MODES = {"cover": resize_center_crop_uint8, "fit": resize_fit_pad_uint8}


class VideoStreamInfo(NamedTuple):
    """What a container header reports about its video stream."""

    height: int
    width: int
    frame_rate: float | None  # None when the header reports nothing usable


def video_stream_info(data: bytes) -> VideoStreamInfo | None:
    """Read a clip's dimensions and frame rate from its container header.

    Demuxing is CPU-side FFmpeg inside PyNvVideoCodec, so this costs no GPU and
    decodes no frame — everything here comes straight off the header, in one
    open, so a caller wanting both does not pay for two.

    The dimensions are the *coded* ones. A container may additionally carry a
    display matrix (a phone shooting portrait usually records landscape frames
    plus a 90-degree rotation); the demuxer does not expose it and the decode
    path does not apply it, so a clip carrying that metadata decodes
    pixel-identically to the same clip without it. Coded dimensions therefore
    describe the frames a caller actually receives, which is what a caller
    sizing its output against them needs.

    Returns ``None`` when the header cannot be read or reports no usable
    dimensions, leaving the caller on its own defaults: this is a convenience
    probe, and a genuinely unreadable stream still fails with a proper error at
    decode.
    """
    try:
        import PyNvVideoCodec as nvc
    except ImportError:
        return None

    position = 0

    def _read(buf: bytearray) -> int:
        nonlocal position
        chunk = data[position : position + len(buf)]
        buf[: len(chunk)] = chunk
        position += len(chunk)
        return len(chunk)

    try:
        demuxer = nvc.CreateDemuxer(_read)
        height, width = int(demuxer.Height()), int(demuxer.Width())
        frame_rate = float(demuxer.FrameRate())
    except nvc.PyNvVCException:
        return None
    if height <= 0 or width <= 0:
        return None
    return VideoStreamInfo(height, width, frame_rate if frame_rate > 0 else None)


def decode_video_reference_window(
    data: bytes,
    *,
    first_frame: int,
    last_frame: int,
    target_h: int,
    target_w: int,
    device: torch.device,
    resize: str = "cover",
    frame_step: int = 1,
) -> torch.Tensor:
    """Decode frames ``[first_frame, last_frame]`` of a reference on device.

    Returns uint8 ``[T, target_h, target_w, 3]``. Indices are Python-style:
    non-negative counts from the start, negative from the end, so ``-1`` is
    the last frame and ``(-8, -1)`` the final eight. Both ends are inclusive.

    ``frame_step`` retains every n-th frame of the range, so ``(0, 96)`` with
    ``frame_step=6`` yields frames 0, 6, ... 96 — seventeen frames, not
    ninety-seven. A caller whose model expects a frame spacing the source was
    not shot at uses this to pick the right frames; the ratio of the two rates
    is the caller's to compute, and no rate is named here. Skipped frames are
    still decoded (inter-frame compression leaves no choice) but are neither
    resized nor retained, so the cost is decode time, not memory. Only
    non-negative ranges may step: the negative form wraps a ring whose length
    is not known until EOS, and combining the two is not supported.

    ``resize`` selects how each frame reaches ``target_h x target_w``:
    ``"cover"`` scales to fill and center-crops (the default, and what video
    continuation wants); ``"fit"`` scales to fit and pads, for references whose
    frame edges carry signal. See :func:`resize_center_crop_uint8` and
    :func:`resize_fit_pad_uint8`.

    A negative index costs a decode to EOS — the memory-buffer demuxer is a
    forward-only feeder, seeking is not assumed — so the caller pays for the
    whole clip when asking from the end. Non-negative ranges stop as soon as
    the range is filled. Frames are resized to the target resolution before
    retention, so a high-resolution source never dominates memory. Clips
    shorter than the range return what exists; the caller pads.

    This decodes what it is asked for and imposes no policy of its own: any
    bound on range size belongs to the model that knows what it can use.
    """
    if (first_frame < 0) != (last_frame < 0):
        raise ValueError(
            f"first_frame and last_frame must both count from the start or "
            f"both from the end, got ({first_frame}, {last_frame})."
        )
    if first_frame > last_frame:
        raise ValueError(
            f"first_frame must not exceed last_frame, got ({first_frame}, {last_frame})."
        )
    if frame_step < 1:
        raise ValueError(f"frame_step must be at least 1, got {frame_step}.")
    if frame_step > 1 and first_frame < 0:
        raise ValueError(
            f"frame_step > 1 is only supported for non-negative ranges, got "
            f"({first_frame}, {last_frame}) with frame_step={frame_step}."
        )
    resize_frames = _RESIZE_MODES.get(resize)
    if resize_frames is None:
        raise ValueError(
            f"Unknown resize mode {resize!r}; expected one of {sorted(_RESIZE_MODES)}."
        )
    window = last_frame - first_frame + 1
    from_end = first_frame < 0
    try:
        import PyNvVideoCodec as nvc
    except ImportError as exc:
        raise ImportError(
            "PyNvVideoCodec is required for video-reference decoding; "
            "install the declared dependency (pip install PyNvVideoCodec)."
        ) from exc

    position = 0

    def _read(buf: bytearray) -> int:
        nonlocal position
        chunk = data[position : position + len(buf)]
        buf[: len(chunk)] = chunk
        position += len(chunk)
        return len(chunk)

    demuxer = None
    decoder = None
    try:
        try:
            # CPU-side FFmpeg demux: failure here means the bytes are not a
            # readable stream — a content problem, not a capacity one.
            demuxer = nvc.CreateDemuxer(_read)
        except nvc.PyNvVCException as exc:
            raise ValueError(
                f"Video reference could not be demuxed (corrupt or not a "
                f"supported container): {exc}"
            ) from exc
        try:
            decoder = nvc.CreateDecoder(
                gpuid=device.index or 0,
                codec=demuxer.GetNvCodecId(),
                usedevicememory=True,
                outputColorType=nvc.OutputColorType.RGB,
            )
        except nvc.PyNvVCException as exc:
            # Init failure on a demuxable stream is genuinely ambiguous — an
            # unsupported codec/profile (client-fixable by re-encoding) and a
            # driver/session failure (deployment fault) raise the same
            # exception type with no inspectable code. Make neither
            # categorical claim: stay unclassified (500), with a message
            # naming both possibilities.
            raise RuntimeError(
                f"NVDEC decoder initialization failed for this stream — the "
                f"codec/profile may be unsupported on this GPU, or the "
                f"decoder session could not be created: {exc}"
            ) from exc

        # Non-negative ranges retain exactly the requested slice, so the ring
        # is filled once; negative ranges cannot know the length up front, so
        # it wraps and holds the trailing `tail` frames until EOS.
        tail = -first_frame if from_end else (window + frame_step - 1) // frame_step
        ring = torch.empty(tail, target_h, target_w, 3, dtype=torch.uint8, device=device)
        count = 0  # frames decoded so far, i.e. the index of the next frame
        kept = 0  # frames written into the ring
        try:
            for packet in demuxer:
                done = False
                for frame in decoder.Decode(packet):
                    if not from_end and count > last_frame:
                        done = True
                        break
                    if from_end or (
                        count >= first_frame and (count - first_frame) % frame_step == 0
                    ):
                        decoded = torch.from_dlpack(frame)
                        # Ownership copy off the NVDEC surface (recycled by
                        # the decoder) and resize-before-retain in one step.
                        ring[kept % tail].copy_(
                            resize_frames(decoded.unsqueeze(0), target_h, target_w)[0]
                        )
                        kept += 1
                    count += 1
                if done:
                    break
        except torch.cuda.OutOfMemoryError as exc:
            raise MemoryError(
                f"Out of device memory while decoding the video reference "
                f"({tail} frames @ {target_w}x{target_h} retained): {exc}"
            ) from exc
        except nvc.PyNvVCException as exc:
            raise ValueError(
                f"Video reference failed to decode (corrupt or unsupported "
                f"stream for this deployment's decoder): {exc}"
            ) from exc

        if count == 0:
            raise ValueError(
                "Video reference contains no decodable frames; the payload "
                "may be corrupt or use an unsupported codec."
            )
        if kept < tail:
            frames = ring[:kept]
        else:
            start = kept % tail
            frames = ring if start == 0 else torch.cat([ring[start:], ring[:start]])
        # `frames` now holds the trailing `tail` frames in order; a negative
        # last_frame other than -1 drops the ones after it.
        return frames[:window] if from_end else frames
    finally:
        del decoder
        del demuxer
