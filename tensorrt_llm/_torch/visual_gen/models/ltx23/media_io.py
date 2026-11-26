# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""Source video and audio decoding for the native LTX-2.3 retake pipeline.

Frames are display-oriented, full-range RGB uint8. Audio retains its original
rate and channel layout on the video's presentation timeline. Model-specific
resizing, resampling and normalization remain in the retake pipeline.
"""

import math
from collections.abc import Iterator
from fractions import Fraction
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import torch

from tensorrt_llm.inputs.multimodal_data import AudioData

from ..ltx2.ltx2_core.types import VideoPixelShape

if TYPE_CHECKING:
    import av


def _rotation(frame: "av.VideoFrame") -> int:
    rotation = float(frame.rotation)
    if not math.isfinite(rotation):
        raise ValueError(f"Video display rotation must be finite: {rotation}.")
    turns = round(rotation / 90)
    if not math.isclose(rotation, turns * 90, abs_tol=1e-3):
        raise ValueError(f"Only quarter-turn video display rotations are supported: {rotation}.")
    return turns % 4


def _validate_transfer(stream: "av.video.stream.VideoStream") -> None:
    # AVFrame transfer tags are unavailable in PyAV 16; the decoder context
    # exposes them. Converting HDR into RGB8 requires explicit tone mapping.
    if stream.codec_context.color_trc in (16, 18):
        raise ValueError("PQ/HLG video requires HDR tone mapping before RGB8 decoding.")


def _rgb_frame(
    frame: "av.VideoFrame",
    reformatter: "av.video.reformatter.VideoReformatter",
    *,
    fallback_colorspace: int = 2,
    fallback_color_range: int = 0,
) -> np.ndarray:
    if frame.format.is_rgb or frame.format.name.startswith("gray"):
        pixels = frame.to_ndarray(format="rgb24")
    else:
        # FFmpeg AVCOL_SPC values differ from SWS_CS values for SMPTE170M.
        colorspace = int(frame.colorspace)
        if colorspace in (0, 2):
            colorspace = fallback_colorspace
        matrix = {1: 1, 4: 4, 5: 5, 6: 5, 7: 7, 9: 9}.get(colorspace)
        if matrix is None:
            if colorspace not in (0, 2):
                raise ValueError(f"Unsupported YUV color matrix: {colorspace}.")
            matrix = 5  # Explicit BT.601 fallback for untagged SDR video.
        color_range = int(frame.color_range) or fallback_color_range
        if color_range == 0:
            color_range = 2 if frame.format.name.startswith("yuvj") else 1
        if color_range not in (1, 2):
            raise ValueError(f"Unsupported YUV color range: {color_range}.")
        # RGB has no YUV destination matrix. Choosing a different coefficient
        # table forces PyAV 16 to apply the source matrix even for full-range YUV.
        destination_matrix = 1 if matrix == 5 else 5
        pixels = reformatter.reformat(
            frame,
            format="rgb24",
            src_colorspace=matrix,
            dst_colorspace=destination_matrix,
            src_color_range=color_range,
            dst_color_range=2,
        ).to_ndarray()
    turns = _rotation(frame)
    if turns:
        pixels = np.rot90(pixels, k=turns, axes=(0, 1))
    return np.ascontiguousarray(pixels)


def get_videostream_metadata(source: str | bytes) -> VideoPixelShape:
    """Read display geometry and count frames when the container omits a count."""
    import av

    with av.open(BytesIO(source) if isinstance(source, bytes) else source) as container:
        if not container.streams.video:
            raise ValueError("Media has no video stream.")
        stream = container.streams.video[0]
        frames = container.decode(stream)
        first = next(frames, None)
        if first is None:
            raise ValueError("Video stream contains no decoded frames.")
        _validate_transfer(stream)
        width, height = first.width, first.height
        if _rotation(first) % 2:
            width, height = height, width
        count = int(stream.frames or 0)
        if not count:
            count = 1 + sum(1 for _ in frames)
        rate = stream.average_rate or stream.guessed_rate
        if rate is None or rate <= 0 or not math.isfinite(float(rate)):
            raise ValueError("Video stream has no positive frame rate.")
        return VideoPixelShape(batch=1, frames=count, height=height, width=width, fps=float(rate))


def decode_video_by_frame(source: str | bytes) -> Iterator[torch.Tensor]:
    """Yield display-oriented RGB uint8 tensors shaped ``(1, H, W, C)``.

    Decode sequentially without seeking or changing the frame rate.
    """
    import av

    with av.open(BytesIO(source) if isinstance(source, bytes) else source) as container:
        if not container.streams.video:
            raise ValueError("Media has no video stream.")
        stream = container.streams.video[0]
        reformatters: dict[tuple[int, int], av.video.reformatter.VideoReformatter] = {}
        for frame in container.decode(stream):
            _validate_transfer(stream)
            colorspace = int(frame.colorspace)
            if colorspace in (0, 2):
                colorspace = stream.codec_context.colorspace
            color_range = int(frame.color_range) or stream.codec_context.color_range
            key = (colorspace, color_range)
            if key not in reformatters:
                reformatters[key] = av.video.reformatter.VideoReformatter()
            pixels = _rgb_frame(
                frame,
                reformatters[key],
                fallback_colorspace=stream.codec_context.colorspace,
                fallback_color_range=stream.codec_context.color_range,
            )
            yield torch.from_numpy(pixels).unsqueeze(0)


def decode_audio_from_file(source: str | bytes) -> AudioData | None:
    """Decode original-rate audio as ``(1, channels, samples)`` on the video timeline.

    Trim negative preroll and declared trailing padding; retain delayed starts
    and interior gaps as silence. Return None if there are no valid audio samples.
    """
    import av

    with av.open(BytesIO(source) if isinstance(source, bytes) else source) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        sample_rate = int(stream.codec_context.sample_rate)
        if sample_rate <= 0:
            raise ValueError("Audio stream has no positive sample rate.")
        origin_stream = container.streams.video[0] if container.streams.video else stream
        if origin_stream.start_time is not None and origin_stream.time_base is not None:
            origin = Fraction(origin_stream.start_time) * origin_stream.time_base
        elif container.start_time is not None:
            origin = Fraction(container.start_time, 1_000_000)
        else:
            origin = Fraction(0)
        valid_end = None
        if stream.duration is not None and stream.time_base is not None:
            stream_start = Fraction(stream.start_time or 0) * stream.time_base
            valid_end = stream_start + stream.duration * stream.time_base
        end_sample = round((valid_end - origin) * sample_rate) if valid_end is not None else None
        resampler = av.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
        timestamp_tolerance = (
            max(0, math.ceil(stream.time_base * sample_rate) - 1)
            if stream.time_base is not None
            else 0
        )
        chunks: list[tuple[int, np.ndarray]] = []
        previous_end = None

        def append(frame: "av.AudioFrame") -> None:
            nonlocal previous_end
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Audio frames must carry presentation timestamps.")
            offset = round((frame.pts * frame.time_base - origin) * sample_rate)
            if previous_end is not None and abs(offset - previous_end) <= timestamp_tolerance:
                offset = previous_end
            if previous_end is not None and offset < previous_end:
                raise ValueError("Audio frame presentation timestamps overlap or run backwards.")
            previous_end = offset + frame.samples
            left = max(0, -offset)
            right = frame.samples if end_sample is None else min(frame.samples, end_sample - offset)
            if right > left:
                samples = frame.to_ndarray()[:, left:right]
                if not np.isfinite(samples).all():
                    raise ValueError("Decoded audio contains non-finite samples.")
                chunks.append((offset + left, samples))

        # Decode sequentially: seeking to zero can drop a valid first AAC frame
        # depending on the demuxer version and edit-list handling.
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Audio frames must carry presentation timestamps.")
            if frame.sample_rate != sample_rate or frame.layout.name != stream.layout.name:
                raise ValueError("Audio rate or channel layout changes within the stream.")
            for converted in resampler.resample(frame):
                append(converted)
        for converted in resampler.resample(None):
            append(converted)
    if not chunks:
        return None
    length = chunks[-1][0] + chunks[-1][1].shape[-1]
    samples = np.zeros((chunks[0][1].shape[0], length), dtype=np.float32)
    for offset, values in chunks:
        samples[:, offset : offset + values.shape[-1]] = values
    return AudioData(samples=torch.from_numpy(samples).unsqueeze(0), sample_rate=sample_rate)
