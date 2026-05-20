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
from typing import List, Optional, Union

import torch

from tensorrt_llm.llmapi.utils import set_api_status


@set_api_status("prototype")
@dataclass
class VisualGenMetrics:
    """Engine-side timing measurements for a single VisualGen request.

    All timings are wall-clock seconds. The three sub-phase numbers are
    measured on the GPU stream via ``torch.cuda.Event(enable_timing=True)``
    records; events are recorded asynchronously so generation does not
    stall. The pipeline synchronizes on the final event before reading
    ``event.elapsed_time``, and that sync cost is amortized into the
    executor-side device→host transfer that already follows the read. The
    sub-phases account for GPU-stream time only, so small host-side work
    shows up as ``generation - (pre_denoise + denoise + post_denoise)``.
    """

    generation: float = 0.0
    """Host wall-clock the executor measured around the engine's inference
    call — what producing the output costs, before any encoding or
    persistence."""

    pre_denoise: float = 0.0
    """GPU-stream time before the denoising loop (text encoding, latent
    prep, conditioning). ``0.0`` if not measured."""

    denoise: float = 0.0
    """GPU-stream time of the denoising loop. For LTX-2's two-stage
    pipeline this covers only the first stage; the second stage rolls into
    ``post_denoise``."""

    post_denoise: float = 0.0
    """GPU-stream time after the denoising loop (VAE decode, format
    conversion, audio decode). ``0.0`` if not measured."""


@set_api_status("prototype")
@dataclass
class VisualGenOutput:
    """Public per-request output from VisualGen.

    Successful outputs populate the media tensor for their modality
    (``image``/``video``/``audio``), the corresponding rate fields, and a
    :class:`VisualGenMetrics` instance. Error outputs leave all media tensors
    and ``metrics`` as ``None`` and set ``error`` to the failure message.

    Use :meth:`save` to persist the output to disk. Pass a single path for
    the ``n == 1`` case (returns :class:`pathlib.Path`) or a list of paths
    for the ``n > 1`` case (returns ``List[Path]``); rates carried on the
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
        path: Union[str, Path, List[Union[str, Path]]],
        *,
        format: Optional[str] = None,
        frame_rate: Optional[float] = None,
        audio_sample_rate: Optional[int] = None,
        quality: int = 95,
    ) -> Union[Path, List[Path]]:
        """Encode this output to disk via :mod:`tensorrt_llm.media.encoding`.

        Args:
            path: Where to write. A single :class:`str`/:class:`pathlib.Path`
                writes one file (batched tensors collapse to the first
                slice); a list of paths writes one file per batch item via
                :func:`~tensorrt_llm.media.encoding.save_images` /
                :func:`~tensorrt_llm.media.encoding.save_videos`. In both
                cases format is inferred from the extension unless
                ``format`` is given.
            format: Explicit format override (``'png'``/``'jpg'``/``'webp'``
                for images, ``'mp4'``/``'avi'`` for video).
            frame_rate: Override the frame rate for video output. Defaults to
                ``self.frame_rate`` when not provided.
            audio_sample_rate: Override the audio sample rate. Defaults to
                ``self.audio_sample_rate`` when not provided.
            quality: Quality for lossy image formats (1-100).

        Returns:
            :class:`pathlib.Path` when ``path`` is a single path, or a list
            of :class:`pathlib.Path` (in batch order) when ``path`` is a list.

        Raises:
            RuntimeError: When the output carries an upstream error.
            ValueError: When video output lacks a frame rate, when the
                output carries no media tensor at all, or when the list
                length does not match the batch size.
            NotImplementedError: When the output is audio-only.
        """
        from tensorrt_llm.media.encoding import save_image, save_images, save_video, save_videos

        if self.error is not None:
            raise RuntimeError(
                f"Cannot save output: request {self.request_id} failed with error: {self.error}"
            )

        is_batch = isinstance(path, list)

        if self.image is not None:
            if is_batch:
                saved_list = save_images(
                    self.image, [str(p) for p in path], format=format, quality=quality
                )
                return [Path(p) for p in saved_list]
            saved = save_image(self.image, path, format=format, quality=quality)
            return Path(saved)

        if self.video is not None:
            fr = frame_rate if frame_rate is not None else self.frame_rate
            if fr is None:
                raise ValueError(
                    "Cannot save video: frame_rate is not set on the output and was not "
                    "provided as a keyword argument."
                )
            asr = audio_sample_rate if audio_sample_rate is not None else self.audio_sample_rate
            asr_value = asr if asr is not None else 24000
            if is_batch:
                saved_list = save_videos(
                    self.video,
                    [str(p) for p in path],
                    audios=self.audio,
                    frame_rate=fr,
                    format=format,
                    audio_sample_rate=asr_value,
                )
                return [Path(p) for p in saved_list]
            saved = save_video(
                self.video,
                path,
                audio=self.audio,
                frame_rate=fr,
                format=format,
                audio_sample_rate=asr_value,
            )
            return Path(saved)

        if self.audio is not None:
            raise NotImplementedError("Saving audio-only outputs is not supported in this release.")

        raise ValueError(
            f"Cannot save output: request {self.request_id} carries no media "
            "(image/video/audio are all None)."
        )
