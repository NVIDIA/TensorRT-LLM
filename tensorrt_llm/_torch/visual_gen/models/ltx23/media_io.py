# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""Source video and audio decoding for the native LTX-2.3 retake pipeline."""

from collections.abc import Iterator

import numpy as np
import torch

from tensorrt_llm.inputs.multimodal_data import AudioData

from ..ltx2.ltx2_core.types import VideoPixelShape


def get_videostream_metadata(path: str) -> VideoPixelShape:
    """Read the first video stream's shape and frame rate with PyAV.

    Container metadata does not always carry a frame count. In that case the
    stream is decoded once to obtain an exact count rather than estimating it
    from duration and frame rate.
    """
    import av

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"media file has no video stream: {path}")
        video_stream = container.streams.video[0]
        if video_stream.average_rate is None:
            raise ValueError(f"video stream has no frame rate: {path}")
        fps = float(video_stream.average_rate)
        num_frames = int(video_stream.frames or 0)
        if num_frames == 0:
            num_frames = sum(1 for _ in container.decode(video_stream))
        return VideoPixelShape(
            batch=1,
            frames=num_frames,
            height=int(video_stream.codec_context.height),
            width=int(video_stream.codec_context.width),
            fps=fps,
        )


def decode_video_by_frame(path: str) -> Iterator[torch.Tensor]:
    """Yield RGB frames by sequential frame index.

    Each yielded tensor has shape ``(1, H, W, C)`` and dtype ``uint8``. Frame
    indices are used instead of presentation timestamps so conditioning input
    selection remains deterministic for retake clips.
    """
    import av

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"media file has no video stream: {path}")
        video_stream = container.streams.video[0]
        for frame in container.decode(video_stream):
            array = frame.to_rgb().to_ndarray()
            yield torch.from_numpy(array).unsqueeze(0)


def decode_audio_from_file(path: str) -> AudioData | None:
    """Decode an audio stream from the beginning.

    Returns an :class:`AudioData` whose samples are ``(1, channels, samples)``, or
    ``None`` when the file carries no audio stream.
    """
    import av

    with av.open(path) as container:
        if not container.streams.audio:
            return None
        audio_stream = container.streams.audio[0]

        container.seek(0, stream=audio_stream)
        sample_rate = None
        resampler = None

        samples = []
        first_frame_time = None
        for frame in container.decode(audio_stream):
            if frame.pts is None:
                continue
            if frame.sample_rate <= 0:
                raise ValueError(
                    f"decoded audio frame has invalid sample rate: {frame.sample_rate}"
                )
            frame_time = float(frame.pts * (frame.time_base or audio_stream.time_base))
            frame_end = frame_time + frame.samples / frame.sample_rate
            if frame_end < 0:
                continue
            if first_frame_time is None:
                first_frame_time = frame_time
                sample_rate = int(frame.sample_rate)
                resampler = av.AudioResampler(format="fltp", rate=sample_rate)
            samples.extend(output.to_ndarray() for output in resampler.resample(frame))
        if resampler is not None:
            samples.extend(output.to_ndarray() for output in resampler.resample(None))

    if not samples:
        return None

    audio = np.concatenate(samples, axis=-1)

    # Discard codec preroll before the video's time origin.
    skip_samples = round(-first_frame_time * sample_rate)
    if skip_samples > 0:
        audio = audio[..., skip_samples:]

    waveform = torch.from_numpy(audio).unsqueeze(0)
    return AudioData(samples=waveform, sample_rate=sample_rate)
