# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public output types for the VisualGen API.

:class:`VisualGenOutput` is the user-facing return value of
:meth:`tensorrt_llm.visual_gen.VisualGen.generate` (and what awaiting a
:class:`tensorrt_llm.visual_gen.VisualGenResult` resolves to).
:class:`VisualGenMetrics` carries engine-side timing measurements; it lives
on every successful output and is ``None`` on error outputs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from tensorrt_llm.llmapi.utils import set_api_status


@set_api_status("prototype")
@dataclass
class VisualGenMetrics:
    """Engine-side timing measurements for a single VisualGen request.

    All four timings are wall-clock milliseconds. ``pipeline_ms`` is measured
    by the executor as host wall-clock around ``pipeline.infer()``. The three
    sub-phase numbers are measured on the GPU stream via
    ``torch.cuda.Event(enable_timing=True)`` records; events are recorded
    asynchronously so the pipeline does not stall, and ``event.elapsed_time``
    performs an implicit sync amortized into the executor-side sync that
    already occurs when the response is consumed. The sub-phases account for
    GPU-stream time only, so small host-side work shows up as
    ``pipeline_ms - (pre_denoise_ms + denoise_ms + post_denoise_ms)``.

    For LTX-2's two-stage pipeline, ``denoise_ms`` covers the first stage and
    the second stage rolls into ``post_denoise_ms``.
    """

    pipeline_ms: float = 0.0
    pre_denoise_ms: float = 0.0
    denoise_ms: float = 0.0
    post_denoise_ms: float = 0.0


@set_api_status("prototype")
@dataclass
class VisualGenOutput:
    """Public per-request output from VisualGen.

    Successful outputs populate the media tensor for their modality
    (``image``/``video``/``audio``), the corresponding rate fields, and a
    :class:`VisualGenMetrics` instance. Error outputs leave all media tensors
    and ``metrics`` as ``None`` and set ``error`` to the failure message.

    Use :meth:`save` to persist the output to disk; rates carried on the
    output are used as defaults and can be overridden via keyword args.
    """

    request_id: int = -1
    image: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    frame_rate: Optional[float] = None
    audio_sample_rate: Optional[int] = None
    error: Optional[str] = None
    metrics: Optional[VisualGenMetrics] = None

    def save(
        self,
        path,
        *,
        format: Optional[str] = None,
        frame_rate: Optional[float] = None,
        audio_sample_rate: Optional[int] = None,
        quality: int = 95,
    ) -> Path:
        """Encode this output to disk via :mod:`tensorrt_llm.media.encoding`.

        Args:
            path: Output file path. Format is inferred from the extension
                unless ``format`` is given.
            format: Explicit format override (``'png'``/``'jpg'``/``'webp'``
                for images, ``'mp4'``/``'avi'`` for video).
            frame_rate: Override the frame rate for video output. Defaults to
                ``self.frame_rate`` when not provided.
            audio_sample_rate: Override the audio sample rate. Defaults to
                ``self.audio_sample_rate`` when not provided.
            quality: Quality for lossy image formats (1-100).

        Returns:
            :class:`pathlib.Path` of the saved file.

        Raises:
            VisualGenError: When the output carries an error, when no media
                tensor is present, or when video output lacks a frame rate.
        """
        # Lazy imports keep the encoding stack and the engine error type out
        # of import time for users who only construct outputs.
        from tensorrt_llm.media.encoding import save_image, save_video
        from tensorrt_llm.visual_gen.visual_gen import VisualGenError

        if self.error is not None:
            raise VisualGenError(
                f"Cannot save output: request {self.request_id} failed with error: {self.error}"
            )

        if self.image is not None:
            saved = save_image(self.image, path, format=format, quality=quality)
            return Path(saved)

        if self.video is not None:
            fr = frame_rate if frame_rate is not None else self.frame_rate
            if fr is None:
                raise VisualGenError(
                    "Cannot save video: frame_rate is not set on the output and was not "
                    "provided as a keyword argument."
                )
            asr = audio_sample_rate if audio_sample_rate is not None else self.audio_sample_rate
            saved = save_video(
                self.video,
                path,
                audio=self.audio,
                frame_rate=fr,
                format=format,
                audio_sample_rate=asr if asr is not None else 24000,
            )
            return Path(saved)

        if self.audio is not None:
            raise VisualGenError("Saving audio-only outputs is not supported in this release.")

        raise VisualGenError(
            f"Cannot save output: request {self.request_id} carries no media "
            "(image/video/audio are all None)."
        )
