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
from typing import Optional
from urllib.parse import urlparse

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
        data = data[comma + 1 :]
    try:
        return base64.b64decode(data, validate=True)
    except ValueError as exc:
        # binascii.Error subclasses ValueError.
        raise ValueError("reference is not valid base64 data.") from exc


def _resolve_reference_string(reference: str) -> bytes:
    """Resolve one reference string to raw bytes, dispatching on URL scheme.

    Mirrors the LLM multimodal loader so serve references accept the same forms:
    ``http(s)`` fetches through the SSRF-guarded loader (private-address block,
    redirect re-validation, timeout, size cap); ``file://`` and bare local paths
    read from disk; ``data:`` and base64 strings decode inline. A bare string is
    decoded as base64 first and, failing that, read as a local file path.
    Fetch/read failures become ``ValueError`` so a bad URL or path is a client
    400, not a server 500.
    """
    scheme = urlparse(reference).scheme
    if scheme in ("http", "https"):
        try:
            return _safe_request_get(reference).content
        except Exception as exc:
            raise ValueError(f"reference URL could not be fetched: {exc}") from exc
    if scheme == "file":
        try:
            return Path(_normalize_file_uri(reference)).read_bytes()
        except OSError as exc:
            raise ValueError(f"reference file could not be read: {exc}") from exc
    if scheme == "data":
        return _read_reference_payload(reference)
    # Bare string: base64 first (the established default), else a local file path
    # so a plain path works without the file:// scheme.
    try:
        return _read_reference_payload(reference)
    except ValueError:
        try:
            return Path(reference).read_bytes()
        except OSError as exc:
            raise ValueError(
                f"reference is not valid base64 data, and not a readable local file: {exc}"
            ) from exc


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
