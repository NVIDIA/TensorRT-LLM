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

import base64
import functools
import math
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from tensorrt_llm.inputs.media_io import BaseMediaIO, _normalize_file_uri


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


class FrameSelector(ABC):
    """Which decoded frames a decode call retains.

    Selection is always stated explicitly — there is no default policy — so a
    caller cannot silently inherit one model's frame convention. ``Window`` is
    the only strategy implemented; fps / whole-clip / explicit-index selection
    are the obvious extensions and are added when a consumer needs one.
    """


@dataclass(frozen=True)
class WindowSelector(FrameSelector):
    """A contiguous inclusive range of frames.

    Indices are Python-style: non-negative counts from the start, negative
    from the end, so ``-1`` is the last frame and ``(-8, -1)`` the final
    eight. Both ends must count from the same end.
    """

    first: int
    last: int

    def __post_init__(self) -> None:
        if (self.first < 0) != (self.last < 0):
            raise ValueError(
                f"first and last must both count from the start or both from "
                f"the end, got ({self.first}, {self.last})."
            )
        if self.first > self.last:
            raise ValueError(f"first must not exceed last, got ({self.first}, {self.last}).")


def _nvdec_decode(
    data: bytes,
    *,
    selector: FrameSelector,
    device: torch.device,
    target_hw: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Decode the frames named by ``selector`` on device, via NVDEC.

    Returns uint8 ``[T, H, W, 3]``, at ``target_hw`` when given and at the
    source resolution otherwise. Resizing happens *before* a frame is
    retained, so a high-resolution source never dominates memory: at any
    instant only one source-resolution frame is alive.

    A negative index costs a decode to EOS — the memory-buffer demuxer is a
    forward-only feeder, seeking is not assumed — so the caller pays for the
    whole clip when asking from the end. Non-negative ranges stop as soon as
    the range is filled. Clips shorter than the range return what exists; the
    caller pads.

    This decodes what it is asked for and imposes no policy of its own: any
    bound on range size belongs to the model that knows what it can use.
    """
    if not isinstance(selector, WindowSelector):
        raise NotImplementedError(f"{type(selector).__name__} is not implemented.")
    first_frame, last_frame = selector.first, selector.last
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
        tail = -first_frame if from_end else window
        # With a target the ring is sized up front, so a range that matches no
        # frame still yields a correctly shaped empty result. Without one, the
        # frame size is unknown until the stream yields a frame.
        ring = (
            torch.empty(tail, *target_hw, 3, dtype=torch.uint8, device=device)
            if target_hw is not None
            else None
        )
        count = 0  # frames decoded so far, i.e. the index of the next frame
        kept = 0  # frames written into the ring
        try:
            for packet in demuxer:
                done = False
                for frame in decoder.Decode(packet):
                    if not from_end and count > last_frame:
                        done = True
                        break
                    if from_end or count >= first_frame:
                        decoded = torch.from_dlpack(frame)
                        # Resize before retaining, so only one frame is ever
                        # alive at the source resolution.
                        retained = (
                            resize_center_crop_uint8(decoded.unsqueeze(0), *target_hw)[0]
                            if target_hw is not None
                            else decoded
                        )
                        if ring is None:
                            ring = torch.empty(
                                tail, *retained.shape, dtype=torch.uint8, device=device
                            )
                        # Ownership copy off the NVDEC surface, which the
                        # decoder recycles.
                        ring[kept % tail].copy_(retained)
                        kept += 1
                    count += 1
                if done:
                    break
        except torch.cuda.OutOfMemoryError as exc:
            at = f" @ {target_hw[1]}x{target_hw[0]}" if target_hw is not None else ""
            raise MemoryError(
                f"Out of device memory while decoding the video reference "
                f"({window} frames{at} retained): {exc}"
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
        if ring is None:
            # Reachable only without a target when the range matched no frame:
            # there is no source size to report, so the empty result carries
            # zero spatial dims.
            return torch.empty(0, 0, 0, 3, dtype=torch.uint8, device=device)
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


class NvdecVideoMediaIO(BaseMediaIO[torch.Tensor]):
    """NVDEC-backed video I/O returning frames on device.

    A sibling of :class:`~tensorrt_llm.inputs.media_io.VideoMediaIO` rather than
    a subclass: that one decodes with cv2 on the CPU and returns ``VideoData``,
    a contract its VLM callers depend on. This one returns a device tensor, so
    it shares the base class and nothing else.

    ``selector`` is required — see :class:`FrameSelector` for why there is no
    default. ``target_hw`` is optional; frames are resized before retention
    when it is given, and kept at the source resolution when it is not.
    """

    def __init__(
        self,
        *,
        selector: FrameSelector,
        device: torch.device,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._selector = selector
        self._device = device
        self._target_hw = target_hw

    def load_bytes(self, data: bytes) -> torch.Tensor:
        return _nvdec_decode(
            data, selector=self._selector, device=self._device, target_hw=self._target_hw
        )

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, url: str) -> torch.Tensor:
        return self.load_bytes(Path(_normalize_file_uri(url)).read_bytes())


def decode_video_reference_window(
    data: bytes,
    *,
    first_frame: int,
    last_frame: int,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode frames ``[first_frame, last_frame]`` of a reference on device.

    Returns uint8 ``[T, target_h, target_w, 3]``. A thin spelling of
    :func:`_nvdec_decode` with a :class:`WindowSelector`, kept because it reads
    better at a call site that only ever wants a window.
    """
    return _nvdec_decode(
        data,
        selector=WindowSelector(first_frame, last_frame),
        device=device,
        target_hw=(target_h, target_w),
    )
