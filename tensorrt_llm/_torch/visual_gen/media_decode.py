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
"""Worker-side video-reference decoding on NVDEC (PyNvVideoCodec).

Encoded reference bytes arrive from the coordinator; each worker rank demuxes
them from memory, decodes on NVDEC, and retains only the conditioning window,
resized to the request's output resolution — so retained memory is bounded by
the request's own output shape and at any instant only one source-resolution
frame is alive.

PyNvVideoCodec is imported function-locally: ``import tensorrt_llm`` on a
CPU-only host must never load the driver-linked extension.
"""

import functools
import math
import os

import torch


class MediaDecodeError(ValueError):
    """Client-class failure: the reference itself is unusable.

    Corrupt/undecodable content, unsupported codec, zero frames, or the
    decode-work limit. Maps to HTTP 400 / a plain ``ValueError`` at the
    public Python boundary.
    """


class VisualGenCapacityError(RuntimeError):
    """Capacity-class failure: a valid request does not fit this deployment.

    CUDA/NVDEC allocation or session-init failure. Maps to HTTP 503; never
    a client error — the input is not malformed.
    """


def classify_worker_error(exc: BaseException) -> str | None:
    """Failure class for the response channel: "client", "capacity", or None.

    Only the two dedicated classes are mapped — a bare ``ValueError`` from a
    model bug must stay an unclassified runtime failure, not become a 400.
    """
    if isinstance(exc, MediaDecodeError):
        return "client"
    if isinstance(exc, (VisualGenCapacityError, torch.cuda.OutOfMemoryError)):
        return "capacity"
    return None


def synchronize_media_prepare_status(exc: Exception | None) -> None:
    """All-rank convergence point between media prepare and model collectives.

    Every rank decodes/prepares its media independently; a rank that failed
    while others proceed into the transformer's collectives would hang the
    job. All ranks call this with their local outcome; if any failed, the
    lowest failing rank's error class + message is broadcast, the failing
    rank(s) re-raise their own exception, and every healthy rank raises a
    reconstructed equivalent in lockstep. Runs on CPU tensors so
    the hybrid (``cpu:gloo``) process group carries it even when the failure
    was CUDA/NVDEC initialization. Converges *caught* failures only — a fatal
    process or context death is beyond its reach.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        if exc is not None:
            raise exc
        return

    healthy_sentinel = 2**31 - 1
    rank = dist.get_rank()
    flag = torch.tensor([rank if exc is not None else healthy_sentinel], dtype=torch.int64)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    failing_rank = int(flag.item())
    if failing_rank == healthy_sentinel:
        return

    payload = [None]
    if rank == failing_rank:
        payload = [(classify_worker_error(exc), str(exc))]
    dist.broadcast_object_list(payload, src=failing_rank)

    if exc is not None:
        raise exc
    kind, message = payload[0]
    message = f"[rank {failing_rank}] {message}"
    if kind == "client":
        raise MediaDecodeError(message)
    if kind == "capacity":
        raise VisualGenCapacityError(message)
    raise RuntimeError(message)


# Decode-work default: 5 min @ 24 fps, far above the canonical ~8 s reference.
# Deliberately its own constant — the serve's output cap (``MAX_VIDEO_FRAMES``)
# is a different policy that happens to share the value today.
DEFAULT_MAX_REFERENCE_DECODE_FRAMES = 7200


def max_reference_decode_frames() -> int | None:
    """Decode-work limit for a video reference, or ``None`` when disabled.

    Bounds serial worker occupancy for forward-only ``keep="last"`` decoding
    (encoded size cannot: hours of low-bitrate video are small on disk). The
    default sits far above the canonical ~8 s reference; trusted deployments
    may raise it or disable it with ``TRTLLM_MAX_REFERENCE_DECODE_FRAMES=0``.
    """
    raw = os.environ.get("TRTLLM_MAX_REFERENCE_DECODE_FRAMES")
    if raw is None:
        return DEFAULT_MAX_REFERENCE_DECODE_FRAMES
    limit = int(raw)
    return None if limit <= 0 else limit


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

    Applied to the worker-decoded reference frames before retention.
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


def decode_video_reference_window(
    data: bytes,
    *,
    window: int,
    keep: str,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode encoded reference bytes into the conditioning window on device.

    Returns a uint8 ``[T, target_h, target_w, 3]`` tensor, ``T <= window``:
    the first ``window`` frames (``keep="first"``, decode stops early) or the
    last ``window`` (``keep="last"``, sequential decode to EOS through a
    preallocated ring — the memory-buffer demuxer is a forward-only feeder,
    seeking is not assumed). Shorter-than-window clips return what exists;
    the pipeline right-pads. Frames are resized to the target resolution
    before retention, so a high-resolution source never dominates memory.
    """
    try:
        import PyNvVideoCodec as nvc
    except ImportError as exc:
        raise ImportError(
            "PyNvVideoCodec is required for video-reference decoding; "
            "install the declared dependency (pip install PyNvVideoCodec)."
        ) from exc

    frame_cap = max_reference_decode_frames()
    if frame_cap is not None and keep == "first" and window > frame_cap:
        raise MediaDecodeError(
            f"Conditioning window of {window} frames exceeds the reference "
            f"decode limit of {frame_cap} (TRTLLM_MAX_REFERENCE_DECODE_FRAMES)."
        )

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
            raise MediaDecodeError(
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

        ring = torch.empty(window, target_h, target_w, 3, dtype=torch.uint8, device=device)
        count = 0
        try:
            for packet in demuxer:
                for frame in decoder.Decode(packet):
                    if frame_cap is not None and count >= frame_cap:
                        raise MediaDecodeError(
                            f"Video reference exceeds the decode limit of "
                            f"{frame_cap} frames "
                            f"(TRTLLM_MAX_REFERENCE_DECODE_FRAMES); trim the "
                            f"clip{' or use condition_video_keep=first' if keep == 'last' else ''}."
                        )
                    decoded = torch.from_dlpack(frame)
                    # Ownership copy off the NVDEC surface (recycled by the
                    # decoder) and resize-before-retain in one step.
                    ring[count % window].copy_(
                        resize_center_crop_uint8(decoded.unsqueeze(0), target_h, target_w)[0]
                    )
                    count += 1
                    if keep == "first" and count >= window:
                        break
                if keep == "first" and count >= window:
                    break
        except torch.cuda.OutOfMemoryError as exc:
            raise VisualGenCapacityError(
                f"Out of device memory while decoding the video reference "
                f"({window} frames @ {target_w}x{target_h} retained): {exc}"
            ) from exc
        except nvc.PyNvVCException as exc:
            raise MediaDecodeError(
                f"Video reference failed to decode (corrupt or unsupported "
                f"stream for this deployment's decoder): {exc}"
            ) from exc

        if count == 0:
            raise MediaDecodeError(
                "Video reference contains no decodable frames; the payload "
                "may be corrupt or use an unsupported codec."
            )
        if count <= window:
            return ring[:count]
        start = count % window
        if start == 0:
            return ring
        return torch.cat([ring[start:], ring[:start]])
    finally:
        del decoder
        del demuxer
