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


def _infer_format_from_path(
    path: Union[str, Path, List[Union[str, Path]]],
) -> Optional[str]:
    """Return the tensor format implied by *path*'s suffix, or ``None``.

    For a list of paths, every entry must share the same recognized
    tensor suffix; mixed or unrecognized suffixes return ``None`` and
    let the image/video encoder dispatch handle them.
    """
    from tensorrt_llm.media.tensor_payload import TENSOR_FORMATS

    def _suffix_format(p) -> Optional[str]:
        suffix = Path(p).suffix
        fmt = suffix[1:] if suffix.startswith(".") else suffix
        return fmt if fmt in TENSOR_FORMATS else None

    if isinstance(path, list):
        if not path:
            return None
        formats = {_suffix_format(p) for p in path}
        return next(iter(formats)) if len(formats) == 1 else None
    return _suffix_format(path)


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
    action: Optional[torch.Tensor] = None
    frame_rate: Optional[float] = None
    audio_sample_rate: Optional[int] = None
    raw_action_dim: Optional[int] = None
    action_mode: Optional[str] = None
    domain_id: Optional[int] = None
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
        """Encode this output to disk.

        Args:
            path: Where to write. A single :class:`str`/:class:`pathlib.Path`
                writes one file (batched tensors collapse to the first
                slice); a list of paths writes one file per batch item.
                Format is inferred from the extension unless ``format``
                is given.
            format: Explicit format. Image encoders: ``"png"``, ``"jpg"``,
                ``"webp"``. Video encoders: ``"mp4"``, ``"avi"``. Tensor
                payloads: ``"safetensors"``, ``"pt"`` — these carry every
                populated modality (image/video/audio) plus scalar
                metadata (frame_rate, audio_sample_rate) in one file.
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
            NotImplementedError: When the output is audio-only and a
                non-tensor format is requested.
        """
        from tensorrt_llm.media.encoding import save_image, save_images, save_video, save_videos
        from tensorrt_llm.media.tensor_payload import is_tensor_format

        if self.error is not None:
            raise RuntimeError(
                f"Cannot save output: request {self.request_id} failed with error: {self.error}"
            )

        is_batch = isinstance(path, list)

        # Tensor formats carry every populated modality in one payload,
        # so the dispatch table for image/video/audio below does not
        # apply. When ``format`` is omitted, infer it from the path
        # suffix so callers using the documented extension convention
        # (``out.safetensors``/``out.pt``) reach the tensor path.
        resolved_format = format if format is not None else _infer_format_from_path(path)
        if is_tensor_format(resolved_format):
            return self._save_tensor_payload(
                path,
                resolved_format,
                is_batch=is_batch,
                frame_rate=frame_rate,
                audio_sample_rate=audio_sample_rate,
            )

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

    def _save_tensor_payload(
        self,
        path: Union[str, Path, List[Union[str, Path]]],
        fmt: str,
        *,
        is_batch: bool,
        frame_rate: Optional[float] = None,
        audio_sample_rate: Optional[int] = None,
    ) -> Union[Path, List[Path]]:
        """Write the safetensors/pt payload for this output to *path*.

        A single path writes one logical output: when the populated
        media tensor is batched the payload corresponds to the first
        item, matching the image/video encoder paths
        (:func:`~tensorrt_llm.media.encoding.save_image` /
        :func:`~tensorrt_llm.media.encoding.save_video`). A list of
        paths writes one payload per batch item by slicing the
        populated tensors along their leading batch axis.

        ``frame_rate`` and ``audio_sample_rate`` override the
        corresponding fields on ``self`` when present, matching the
        encoder path's override semantics.
        """
        from tensorrt_llm.media.tensor_payload import (
            infer_batch_size,
            save_visual_gen_output_payload,
        )

        batch_size = infer_batch_size(self)

        if not is_batch:
            if batch_size > 1:
                raise ValueError(
                    f"save received a single path but the output carries a batched "
                    f"tensor of size {batch_size}; pass a list of {batch_size} paths "
                    "(one per item)."
                )
            slice_index = 0 if batch_size > 0 else None
            return save_visual_gen_output_payload(
                self,
                path,
                fmt,
                batch_index=slice_index,
                frame_rate=frame_rate,
                audio_sample_rate=audio_sample_rate,
            )

        if len(path) != batch_size:
            raise ValueError(
                f"Number of paths ({len(path)}) does not match batch size ({batch_size})."
            )
        return [
            save_visual_gen_output_payload(
                self,
                p,
                fmt,
                batch_index=i,
                frame_rate=frame_rate,
                audio_sample_rate=audio_sample_rate,
            )
            for i, p in enumerate(path)
        ]

    def _save_bytes(
        self,
        format: str,
        *,
        batch_index: Optional[int] = None,
        frame_rate: Optional[float] = None,
        audio_sample_rate: Optional[int] = None,
    ) -> bytes:
        """Serialize this output to bytes for in-memory transport.

        Internal counterpart to :meth:`save`. The public output API
        exposes only :meth:`save` (file-based); the in-memory bytes
        path is reserved for trtllm-serve's ``b64_json`` transport,
        which derives ``batch_index`` from
        :func:`tensorrt_llm.media.tensor_payload.infer_batch_size`
        before iterating. Only tensor formats are supported today.
        """
        from tensorrt_llm.media.tensor_payload import is_tensor_format, serialize_visual_gen_output

        if self.error is not None:
            raise RuntimeError(
                f"Cannot save output: request {self.request_id} failed with error: {self.error}"
            )
        if not is_tensor_format(format):
            raise ValueError(f"_save_bytes supports only tensor formats today; got {format!r}.")
        return serialize_visual_gen_output(
            self,
            format,
            batch_index=batch_index,
            frame_rate=frame_rate,
            audio_sample_rate=audio_sample_rate,
        )
