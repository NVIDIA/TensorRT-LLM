# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Internal pipeline output dataclass for visual generation models.

This module defines :class:`PipelineOutput`, the dataclass each
``BasePipeline.infer()`` returns. It is internal and serialized over
ZMQ to the client process where it is converted to the public
:class:`tensorrt_llm.visual_gen.VisualGenOutput`.

The :class:`CudaPhaseTimer` helper records ``torch.cuda.Event(enable_timing=True)``
markers at the three phase boundaries (pre-denoise / denoise / post-denoise)
without adding host syncs in the hot path; the implicit sync from
``event.elapsed_time`` is amortized into the executor-side sync that already
occurs when the response is consumed.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.executor import DiffusionResponse
    from tensorrt_llm.visual_gen.output import VisualGenOutput


@dataclass
class PipelineOutput:
    """Internal per-pipeline output.

    Each pipeline ``infer()`` populates the media tensor it produces plus
    the metadata it owns (``frame_rate``, ``audio_sample_rate``) and the
    three CUDA-event-measured timing phases that decompose ``pipeline.infer()``.

    Attributes:
        image: Generated image as ``torch.Tensor`` shape ``(H, W, C)`` or
            ``(B, H, W, C)``, dtype ``uint8``. Populated by Flux pipelines.
        video: Generated video as ``torch.Tensor`` shape ``(T, H, W, C)`` or
            ``(B, T, H, W, C)``, dtype ``uint8``. Populated by Wan and LTX-2.
        audio: Generated audio as ``torch.Tensor``, dtype ``float32``.
            Populated by LTX-2.
        frame_rate: Video frame rate in fps. Populated by video pipelines
            (Wan T2V/I2V emit ``16.0``; LTX-2 emits ``params.frame_rate``).
            ``None`` for image-only pipelines.
        audio_sample_rate: Audio sample rate in Hz. Populated by LTX-2 from
            its audio config (no hard-coded literal). ``None`` for pipelines
            without audio.
        pre_denoise_ms: Wall-clock GPU-stream time before the denoising loop
            (text encoding, latent prep, conditioning), measured by CUDA
            events. ``0.0`` if not measured.
        denoise_ms: Wall-clock GPU-stream time of the denoising loop,
            measured by CUDA events. For LTX-2's two-stage pipeline this
            tracks only the first stage; the second stage rolls into
            ``post_denoise_ms``.
        post_denoise_ms: Wall-clock GPU-stream time after the denoising loop
            (VAE decode, format conversion, audio decode), measured by CUDA
            events. ``0.0`` if not measured.
    """

    image: Optional[torch.Tensor] = None
    video: Optional[torch.Tensor] = None
    audio: Optional[torch.Tensor] = None
    frame_rate: Optional[float] = None
    audio_sample_rate: Optional[int] = None
    pre_denoise_ms: float = 0.0
    denoise_ms: float = 0.0
    post_denoise_ms: float = 0.0


class CudaPhaseTimer:
    """Record ``torch.cuda.Event`` markers at the three pipeline phase boundaries.

    Usage from a pipeline's ``forward()``::

        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        # ...text encoding, latent prep, conditioning...
        timer.mark_denoise_start()
        # ...denoising loop...
        timer.mark_post_start()
        # ...VAE decode, format conversion...
        timer.mark_end()
        return timer.fill(PipelineOutput(...))

    On non-CUDA runs (CPU only) the markers are no-ops and the timing fields
    on ``PipelineOutput`` keep their ``0.0`` defaults.

    Timing methodology: the helper does not call ``torch.cuda.synchronize`` or
    ``Event.synchronize`` itself. ``Event.elapsed_time`` performs whatever
    implicit sync is required when ``fill`` is invoked, and that cost is
    amortized into the post-decode CPU transfer the pipeline already performs
    before returning.
    """

    def __init__(self) -> None:
        if torch.cuda.is_available():
            self._enabled = True
            mk = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
            self._pre_start = mk()
            self._denoise_start = mk()
            self._post_start = mk()
            self._end = mk()
        else:
            self._enabled = False

    def mark_pre_start(self) -> None:
        if self._enabled:
            self._pre_start.record()

    def mark_denoise_start(self) -> None:
        if self._enabled:
            self._denoise_start.record()

    def mark_post_start(self) -> None:
        if self._enabled:
            self._post_start.record()

    def mark_end(self) -> None:
        if self._enabled:
            self._end.record()

    def fill(self, output: "PipelineOutput") -> "PipelineOutput":
        """Populate the three ``*_ms`` fields on ``output``; safe on non-CUDA."""
        if not self._enabled:
            return output
        output.pre_denoise_ms = float(self._pre_start.elapsed_time(self._denoise_start))
        output.denoise_ms = float(self._denoise_start.elapsed_time(self._post_start))
        output.post_denoise_ms = float(self._post_start.elapsed_time(self._end))
        return output


def to_visual_gen_output(resp: "DiffusionResponse") -> "VisualGenOutput":
    """Convert an internal :class:`DiffusionResponse` into a public :class:`VisualGenOutput`.

    On error, ``image``/``video``/``audio``/``metrics`` are left at their
    defaults and ``error`` carries ``resp.error_msg``. On success, all media
    and rate fields are taken from ``resp.output`` (a :class:`PipelineOutput`)
    and ``metrics`` carries the four timings.
    """
    from tensorrt_llm.visual_gen.output import VisualGenMetrics, VisualGenOutput

    if resp.error_msg is not None:
        return VisualGenOutput(request_id=resp.request_id, error=resp.error_msg)
    out = resp.output
    metrics = VisualGenMetrics(
        pipeline_ms=resp.pipeline_ms,
        pre_denoise_ms=out.pre_denoise_ms,
        denoise_ms=out.denoise_ms,
        post_denoise_ms=out.post_denoise_ms,
    )
    return VisualGenOutput(
        request_id=resp.request_id,
        image=out.image,
        video=out.video,
        audio=out.audio,
        frame_rate=out.frame_rate,
        audio_sample_rate=out.audio_sample_rate,
        metrics=metrics,
    )


def split_visual_gen_output(resp: "DiffusionResponse", batch_size: int) -> List["VisualGenOutput"]:
    """Fan out a batched :class:`DiffusionResponse` into per-item outputs.

    On error, returns ``batch_size`` outputs each carrying ``resp.error_msg``
    so the caller can iterate and check ``out.error`` per item. On success,
    slices each present media tensor along dim 0 to produce per-item
    unbatched tensors and shares the metrics object across items (a single
    batched inference produced one set of timings).

    Slicing uses tensor views by default; switch to ``.clone().contiguous()``
    if a downstream consumer reports an aliasing issue.
    """
    from tensorrt_llm.visual_gen.output import VisualGenMetrics, VisualGenOutput

    if resp.error_msg is not None:
        return [
            VisualGenOutput(request_id=resp.request_id, error=resp.error_msg)
            for _ in range(batch_size)
        ]
    out = resp.output
    metrics = VisualGenMetrics(
        pipeline_ms=resp.pipeline_ms,
        pre_denoise_ms=out.pre_denoise_ms,
        denoise_ms=out.denoise_ms,
        post_denoise_ms=out.post_denoise_ms,
    )
    results: List["VisualGenOutput"] = []
    for i in range(batch_size):
        results.append(
            VisualGenOutput(
                request_id=resp.request_id,
                image=out.image[i] if out.image is not None else None,
                video=out.video[i] if out.video is not None else None,
                audio=out.audio[i] if out.audio is not None else None,
                frame_rate=out.frame_rate,
                audio_sample_rate=out.audio_sample_rate,
                metrics=metrics,
            )
        )
    return results
