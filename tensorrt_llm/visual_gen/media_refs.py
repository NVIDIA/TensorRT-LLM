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
"""Reference-media resolution, shared by serve and engine.

Every declared wire form resolves to raw bytes, the canonical form carried all
the way to the pipeline. Used by both the serve boundary
(``tensorrt_llm/serve``) and the engine frontend (``VisualGen.generate_async``),
so this lives here rather than under ``serve`` to avoid an engine -> serve
import.
"""

from __future__ import annotations

import base64
from pathlib import Path
from stat import S_ISREG
from typing import Any

from tensorrt_llm.inputs.media_io import (
    _MAX_RESPONSE_BYTES,
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


def _safe_read_local_file(reference: str) -> bytes:
    """Read a ``path`` reference, bounding what an unlucky path can cost.

    The counterpart of :func:`_safe_request_get` for the local branch. A
    remote caller naming the path is the case worth defending: ``read_bytes``
    on a character device or a FIFO never returns, so an unbounded read is a
    denial of service rather than a bad request. Requiring a regular file
    within the same size cap the remote fetch uses keeps both branches to one
    rule.

    This bounds cost, not reach: any regular file the server process can read
    is still readable. Restricting *which* files a remote caller may name is a
    deployment-policy question, and belongs with the deployment.
    """
    path = Path(_normalize_file_uri(reference))
    try:
        stat = path.stat()  # follows symlinks, so a link to a device is caught
    except OSError as exc:
        raise ValueError(f"reference file could not be read: {exc}") from exc

    if not S_ISREG(stat.st_mode):
        raise ValueError(
            f"reference path is not a regular file: {reference!r}. Character "
            "devices, FIFOs and directories cannot be read as media."
        )
    if stat.st_size > _MAX_RESPONSE_BYTES:
        raise ValueError(
            f"reference file is {stat.st_size} bytes, over the {_MAX_RESPONSE_BYTES}-byte limit."
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ValueError(f"reference file could not be read: {exc}") from exc


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
        return _safe_read_local_file(content)
    if content_format == "base64":
        return _read_reference_payload(content)
    raise ValueError(f"unsupported reference format: {content_format!r}")


def _validate_reference_payload(payload: bytes, *, modality: str) -> None:
    """Reject a payload whose container does not match the declared modality.

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
    elif modality == "audio":
        if sniff_media_kind(payload) != "audio":
            raise ValueError(
                "audio_reference is not a recognized audio container; supported "
                "inputs are WAV/MP3/FLAC/OGG/M4A/AAC."
            )


def prepare_reference_slots(params: Any) -> None:
    """Resolve every reference to raw bytes, in place.

    The single reference choke point, used by the engine (``generate_async``)
    so serve and the standalone Python API share one path. Dispatch is on each
    reference's declared ``format``, never on the shape of its content: the
    declared form is resolved to bytes, content-validated against the slot's
    modality, and written back with ``format`` set to ``"bytes"``.

    ``format`` is rewritten alongside ``content`` because the mutated params
    object is what gets broadcast to the workers; a stale format would tell a
    worker it is holding base64 when it is holding raw bytes.

    Bytes are the canonical form all the way to the pipeline, so a reference
    never touches the filesystem: there is nothing to clean up afterwards, and
    a worker needs no shared filesystem to read what the coordinator resolved.

    Runs before the coordinator broadcasts the request, so a bad reference
    raises ``ValueError`` synchronously and serve keeps its immediate 400.
    """
    for slot in ("image_reference", "video_reference", "audio_reference"):
        modality = slot.split("_", 1)[0]
        for ref in getattr(params, slot, None) or []:
            data = _resolve_reference(ref.content, ref.format)
            _validate_reference_payload(data, modality=modality)
            ref.content = data
            ref.format = "bytes"
