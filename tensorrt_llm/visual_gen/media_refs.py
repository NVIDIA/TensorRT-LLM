# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Reference-media resolve / materialize / cleanup, shared by serve and engine.

Verb convention: ``resolve`` -> bytes, ``materialize`` -> path. These are used
by both the serve boundary (``tensorrt_llm/serve``) and the engine frontend
(``VisualGen.generate_async``), so they live here rather than under ``serve`` to
avoid an engine -> serve import.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Optional

from tensorrt_llm.inputs.media_io import (
    _normalize_file_uri,
    _safe_request_get,
    is_isobmff_image_bytes,
    sniff_media_kind,
)


def _read_reference_payload(reference: str) -> bytes:
    """Decode one base64 (optionally ``data:`` URI) reference string to bytes.

    Payload size is deliberately not checked here: encoded size is not part
    of the request-validity contract, and body limits belong to the
    proxy/ASGI deployment layer (HTTP 413). Base64 decodes strictly so
    malformed encodings — not sizes — are rejected.
    """
    data = reference
    if data.startswith("data:"):
        comma = data.find(",")
        if comma == -1:
            raise ValueError("reference data: URI is malformed (missing comma).")
        # Match the LLM loader: only base64 payloads are supported, and saying so
        # beats letting a percent-encoded body fail as "not valid base64".
        if "base64" not in data[:comma].split(";")[1:]:
            raise ValueError("only base64 data: URIs are supported for references.")
        data = data[comma + 1 :]
    try:
        return base64.b64decode(data, validate=True)
    except ValueError as exc:
        # binascii.Error subclasses ValueError.
        raise ValueError("reference is not valid base64 data.") from exc


def _local_path(reference: str) -> Path:
    """Normalize a ``path`` reference (bare or ``file://``) to a ``Path``."""
    return Path(_normalize_file_uri(reference))


def _resolve_reference(content: Any, content_format: str) -> bytes:
    """Resolve one reference to raw bytes using its declared wire form.

    Dispatch is on the caller-declared ``format``, never on the shape of the
    value: a bare string is otherwise ambiguous between a local path and
    base64, and guessing lets a mistyped path become base64 (or a malformed
    base64 become a filesystem read). Fetch/read/decode failures become
    ``ValueError`` so a bad reference is a client 400, not a server 500.
    """
    if content_format == "bytes":
        if not isinstance(content, bytes):
            raise ValueError(
                f"format='bytes' requires bytes content, got {type(content).__name__}."
            )
        return content
    if not isinstance(content, str):
        raise ValueError(
            f"format={content_format!r} requires string content, got {type(content).__name__}."
        )
    if content_format == "url":
        try:
            return _safe_request_get(content).content
        except Exception as exc:
            raise ValueError(f"reference URL could not be fetched: {exc}") from exc
    if content_format == "path":
        try:
            return _local_path(content).read_bytes()
        except OSError as exc:
            raise ValueError(f"reference file could not be read: {exc}") from exc
    if content_format == "base64":
        return _read_reference_payload(content)
    raise ValueError(f"unsupported reference format: {content_format!r}")


def _materialize_reference(
    payload: bytes, *, modality: str, ref_id: str, media_storage_path: Optional[str]
) -> str:
    """Content-validate a reference payload and persist it, returning its path.

    HEIF/AVIF images are rejected on signature alone (Pillow support depends
    on optional plugins the worker need not share). Video acceptance beyond the
    container signature happens in the worker's NVDEC demux.
    """
    if modality == "image":
        if sniff_media_kind(payload) != "image":
            raise ValueError(
                "image_reference is not a recognized image; supported inputs are PNG/JPEG."
            )
        if is_isobmff_image_bytes(payload):
            raise ValueError(
                "image_reference is a HEIF/AVIF image, which is not a supported "
                "reference format; convert it to PNG or JPEG."
            )
    elif modality == "video":
        if sniff_media_kind(payload) != "video":
            raise ValueError(
                "video_reference is not a recognized media container; supported "
                "inputs are MP4/AVI video."
            )
    # audio: no signature sniffing (sniff_media_kind detects only image/video);
    # the consuming pipeline validates the audio codec in its worker.
    if media_storage_path is None:
        raise ValueError(f"media_storage_path is required to store the {modality}_reference.")
    ref_path = os.path.join(media_storage_path, ref_id)
    with open(ref_path, "wb") as f:
        f.write(payload)
    return ref_path


def cleanup_reference_files(media_storage_path: Optional[str], request_id: str) -> None:
    """Remove the materialized reference inputs for one request.

    References are materialized as ``{request_id}_{modality}_ref_{i}`` (and the
    deprecated ``{request_id}_input_ref``) under ``media_storage_path``. They are
    input-only — unneeded once the pipeline has consumed them — so the request
    owner removes them by the ``request_id`` prefix, covering image/video/audio
    and the deprecated single reference regardless of count. Output files
    (``{request_id}_{i}.<ext>``) carry no ``ref`` and are left untouched.
    Best-effort: already-removed files are ignored.
    """
    if media_storage_path is None:
        return
    for path in Path(media_storage_path).glob(f"{request_id}_*ref*"):
        try:
            path.unlink()
        except OSError:
            pass


def resolve_media_storage_path() -> Path:
    """Resolve the media storage directory, creating it if needed.

    Reads ``TRTLLM_MEDIA_STORAGE_PATH`` (default ``/tmp/trtllm_generated``),
    shared by the serve boundary and the engine so both write materialized
    references to the same place.
    """
    path = Path(os.getenv("TRTLLM_MEDIA_STORAGE_PATH", "/tmp/trtllm_generated"))  # nosec B108
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_reference_slots(
    params: Any, *, request_id: str, media_storage_path: Optional[str]
) -> None:
    """Resolve + materialize each reference to a local path, in place.

    The single reference choke point, used by the engine (``generate_async``)
    so serve and the standalone Python API share one path. Dispatch is on each
    reference's declared ``format``, never on the shape of its content. A
    ``path`` reference is the caller's own file: it passes through — not
    materialized, not cleaned up — with a ``file://`` URI normalized to a plain
    path so the pipeline, which opens paths, can read it. Every other form
    (``url`` / ``base64`` / ``bytes``) resolves to bytes and materializes to
    ``media_storage_path``; those files are reclaimed by
    :func:`cleanup_reference_files` keyed on ``request_id``.

    ``format`` is rewritten alongside ``content``: the mutated params object is
    what gets broadcast to the workers, so a stale format would send e.g.
    ``base64`` to a worker holding a filesystem path.

    Runs before the coordinator broadcasts the request, so a bad reference
    raises ``ValueError`` synchronously (serve keeps its immediate 400). If a
    later slot fails mid-materialize, the files earlier slots wrote are
    reclaimed here so a rejected request leaves nothing on disk.
    """
    try:
        for slot in ("image_reference", "video_reference", "audio_reference"):
            modality = slot.split("_", 1)[0]
            for i, ref in enumerate(getattr(params, slot, None) or []):
                if ref.format == "path":
                    path = _local_path(ref.content)
                    if not path.exists():
                        raise ValueError(f"reference file does not exist: {ref.content}")
                    ref.content = str(path)
                    continue
                data = _resolve_reference(ref.content, ref.format)
                ref.content = _materialize_reference(
                    data,
                    modality=modality,
                    ref_id=f"{request_id}_{modality}_ref_{i}",
                    media_storage_path=media_storage_path,
                )
                ref.format = "path"
    except Exception:
        # The terminal on_finish hook is not wired yet (the request is never
        # enqueued on failure), so reclaim any files earlier slots wrote here.
        cleanup_reference_files(media_storage_path, request_id)
        raise
