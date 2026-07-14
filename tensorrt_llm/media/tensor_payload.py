# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-format serializers for :class:`VisualGenOutput`.

Two payload formats are supported:

- ``"safetensors"``: writes a single file with named tensors
  (``image``/``video``/``audio``). Scalar metadata (``frame_rate``,
  ``audio_sample_rate``) is stored two ways: as a 0-d tensor under
  the same key (so ``safetensors.torch.load(bytes)`` returns it
  alongside the media tensors — consumers call ``.item()`` to
  unbox) and as a stringified value in the file header (preserved
  for callers using ``safe_open(...).metadata()``). No pickle on
  load.
- ``"pt"``: writes a single file via :func:`torch.save` with the
  same tensor keys plus scalar metadata as native Python values.
  Clients should load with ``torch.load(buf, weights_only=True)``
  on PyTorch 2.4+.

Both serializers share the same logical payload shape so the
serve layer can pick the bytes-based path (``b64_json`` transport)
or the file-based path (``url`` transport) without having to
reconstruct the payload twice.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.output import VisualGenOutput


# Tokens recognized by the tensor-payload path. Anything outside this
# set is treated as a media-encoder request by :meth:`VisualGenOutput.save`.
TENSOR_FORMATS = frozenset({"safetensors", "pt"})


def is_tensor_format(fmt: Optional[str]) -> bool:
    """Return True when *fmt* names a tensor payload (safetensors / pt)."""
    return fmt in TENSOR_FORMATS


# Ranks at which each modality is batched. An image tensor with the
# canonical ``(H, W, C)`` shape is unbatched at rank 3 and batched at
# rank 4; video is unbatched at rank 4 ``(T, H, W, C)`` and batched at
# rank 5; audio is unbatched at rank 2 ``(channels, T_audio)`` and
# batched at rank 3. The serializer uses these to decide whether a
# media tensor has a true batch axis to slice along.
_BATCHED_RANKS: Dict[str, int] = {
    "image": 4,
    "video": 5,
    "audio": 3,
}


def _modalities(output: "VisualGenOutput") -> Tuple[Tuple[str, Optional[torch.Tensor]], ...]:
    return (
        ("image", output.image),
        ("video", output.video),
        ("audio", output.audio),
    )


def infer_batch_size(output: "VisualGenOutput") -> int:
    """Return the leading batch dimension across the populated media tensors.

    Image is batched only at rank 4, video at rank 5, audio at rank 3.
    An unbatched media tensor reports a batch size of 1 so list-path
    callers can still ask for ``[0]`` and get a single-item payload.
    Raises :class:`ValueError` when *output* carries no media tensor.
    """
    sizes = set()
    have_media = False
    for name, tensor in _modalities(output):
        if tensor is None:
            continue
        have_media = True
        if tensor.dim() == _BATCHED_RANKS[name]:
            sizes.add(int(tensor.shape[0]))
        else:
            sizes.add(1)
    if not have_media:
        raise ValueError(
            f"Cannot infer batch size: request {output.request_id} carries no media tensor."
        )
    if len(sizes) > 1:
        raise ValueError(
            f"Inconsistent batch sizes across modalities: {sorted(sizes)}. "
            "All populated media tensors must agree on the leading batch axis."
        )
    return next(iter(sizes))


def _collect_tensors_and_metadata(
    output: "VisualGenOutput",
    batch_index: Optional[int],
    *,
    frame_rate_override: Optional[float] = None,
    audio_sample_rate_override: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Build the ``{name: tensor}`` and ``{name: scalar}`` views over *output*.

    When ``batch_index`` is provided, slice each populated media tensor
    along its leading batch dimension *only if* the tensor's rank says
    it actually has a batch axis (see :data:`_BATCHED_RANKS`). Tensors
    that are already unbatched are passed through unchanged so callers
    don't accidentally slice height (image) or frame (video) axes.

    When ``batch_index`` is ``None`` the tensors are written as-is.

    ``frame_rate_override`` / ``audio_sample_rate_override`` take
    precedence over the corresponding fields on *output*. This mirrors
    the video-encoder path's ``frame_rate``/``audio_sample_rate``
    keyword overrides on :meth:`VisualGenOutput.save` so callers that
    pass overrides (because the output fields are ``None`` or need to
    be replaced) get those values in the serialized payload's
    metadata.
    """
    tensors: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, Any] = {}

    for name, src in _modalities(output):
        if src is None:
            continue
        if batch_index is not None and src.dim() == _BATCHED_RANKS[name]:
            sliced = src[batch_index]
        else:
            sliced = src
        tensors[name] = sliced.contiguous().cpu()

    frame_rate = frame_rate_override if frame_rate_override is not None else output.frame_rate
    audio_sample_rate = (
        audio_sample_rate_override
        if audio_sample_rate_override is not None
        else output.audio_sample_rate
    )
    if frame_rate is not None:
        metadata["frame_rate"] = float(frame_rate)
    if audio_sample_rate is not None:
        metadata["audio_sample_rate"] = int(audio_sample_rate)

    return tensors, metadata


def serialize_visual_gen_output(
    output: "VisualGenOutput",
    fmt: str,
    *,
    batch_index: Optional[int] = None,
    frame_rate: Optional[float] = None,
    audio_sample_rate: Optional[int] = None,
) -> bytes:
    """Serialize *output* to in-memory bytes using *fmt*.

    Args:
        output: The :class:`VisualGenOutput` to serialize. Must have at
            least one populated media tensor.
        fmt: ``"safetensors"`` or ``"pt"``.
        batch_index: When set, slice each populated tensor along its
            leading batch dimension before serialization so the result
            corresponds to a single batch item.
        frame_rate: Override the ``frame_rate`` written into the
            payload metadata. Falls back to ``output.frame_rate``.
        audio_sample_rate: Override the ``audio_sample_rate`` written
            into the payload metadata. Falls back to
            ``output.audio_sample_rate``.

    Returns:
        Serialized bytes ready for ``b64_json`` transport or for writing
        to disk via :func:`save_visual_gen_output_payload`.

    Raises:
        ValueError: When *fmt* is not a supported tensor token or
            *output* carries no media tensor.
    """
    if not is_tensor_format(fmt):
        raise ValueError(
            f"Unsupported tensor format: {fmt!r}. Use one of {sorted(TENSOR_FORMATS)}."
        )

    tensors, metadata = _collect_tensors_and_metadata(
        output,
        batch_index,
        frame_rate_override=frame_rate,
        audio_sample_rate_override=audio_sample_rate,
    )
    if not tensors:
        raise ValueError(
            f"Cannot serialize output: request {output.request_id} carries no media tensor."
        )

    if fmt == "safetensors":
        from safetensors.torch import save as safetensors_save

        # Store each scalar twice: as a 0-d tensor (survives the canonical
        # ``safetensors.torch.load(bytes)`` path so consumers can read
        # ``loaded["frame_rate"].item()`` directly) and as a string in the
        # file header (preserved for callers that already use
        # ``safe_open(...).metadata()``). The two views always agree.
        scalar_tensors = {k: torch.as_tensor(v) for k, v in metadata.items()}
        return safetensors_save(
            {**tensors, **scalar_tensors},
            metadata={k: str(v) for k, v in metadata.items()},
        )

    payload: Dict[str, Any] = {**tensors, **metadata}
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def save_visual_gen_output_payload(
    output: "VisualGenOutput",
    path: Union[str, Path],
    fmt: str,
    *,
    batch_index: Optional[int] = None,
    frame_rate: Optional[float] = None,
    audio_sample_rate: Optional[int] = None,
) -> Path:
    """Write the tensor payload for *output* to *path* using *fmt*.

    The path's suffix is normalized to ``.safetensors`` or ``.pt`` when
    missing so on-disk artifacts are always identifiable by extension.
    ``frame_rate`` and ``audio_sample_rate`` override the
    corresponding fields on *output* in the serialized payload's
    metadata, matching :meth:`VisualGenOutput.save`'s encoder path.
    """
    target = Path(path)
    if target.suffix == "":
        target = target.with_suffix(f".{fmt}")
    target.parent.mkdir(parents=True, exist_ok=True)
    data = serialize_visual_gen_output(
        output,
        fmt,
        batch_index=batch_index,
        frame_rate=frame_rate,
        audio_sample_rate=audio_sample_rate,
    )
    target.write_bytes(data)
    return target
