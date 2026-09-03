# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Benchmark online serving throughput for VisualGen (image/video generation).

On the server side, run:
    trtllm-serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --visual_gen_args <config.yaml>

On the client side, run:
    python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
        --workload wan22-t2v-a14b.yaml \
        --max-concurrency 1 \
        --save-result --save-detailed

Generation parameters live in the ``--workload`` document rather than on the CLI.
Its format and the metrics reported are documented in
``examples/visual_gen/serve/BENCHMARKING.md``.
"""

import argparse
import asyncio
import base64
import gc
import json
import math
import os
import sys
import time
import traceback
from argparse import ArgumentParser as FlexibleArgumentParser
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, get_args

import aiohttp
import numpy as np
import yaml
from pydantic import ValidationError
from tqdm.asyncio import tqdm

from tensorrt_llm.serve.visual_gen_metrics import (
    SERVER_TIMING_HEADER,
    VISUAL_GEN_DENOISE_TIMING,
    VISUAL_GEN_GENERATION_TIMING,
    VISUAL_GEN_TOTAL_TIMING,
)
from tensorrt_llm.visual_gen.params import VisualGenParams

VIDEO_BACKEND = "openai-videos"
BACKEND_ENDPOINTS = {
    "openai-images": "/v1/images/generations",
    "openai-image-edits": "/v1/images/edits",
    VIDEO_BACKEND: "/v1/videos",
}
MODALITY_BY_BACKEND = {
    "openai-images": "image",
    "openai-image-edits": "image",
    VIDEO_BACKEND: "video",
}

WORKLOAD_DOC_KEYS = ("backend", "common_params", "requests")
# Reference slots the loader resolves itself, so the document can name a local file
# and the read happens once, before the run. VisualGenParams rejects a bare path.
REFERENCE_KEYS = ("image_reference", "video_reference")
# Input-layer keys: accepted in both layers, but not VisualGenParams fields.
# Input keys a request and common_params both take. A reference takes neither: it
# is the input to one generation, so it belongs to the request it conditions.
COMMON_INPUT_KEYS = ("prompt", "prompt_file")


def _scalar_param_fields() -> dict[str, type]:
    """``VisualGenParams`` fields one CLI value can carry, by field name.

    Derived rather than listed so a flag can never name a field differently
    from the document, and a new scalar field is addressable without an edit.
    """
    fields = {}
    for name, spec in VisualGenParams.model_fields.items():
        inner = [a for a in get_args(spec.annotation) if a is not type(None)]
        inner = inner or [spec.annotation]
        if len(inner) == 1 and inner[0] in (int, float, str):
            fields[name] = inner[0]
    return fields


SCALAR_PARAM_FIELDS = _scalar_param_fields()
# One flag per common_params key, plus the two that carry a whole part of the document.
CLI_KEYS = (*SCALAR_PARAM_FIELDS, *COMMON_INPUT_KEYS, "extra_params", "requests")

TABLE_WIDTH = 70
STAT_COLUMNS = ("mean", "median", "std", "min", "max")
SERVER_TIMING_FIELDS = ("server_e2e", "server_gen", "server_denoise")

PATH_DISABLED_HINT = (
    "The server refuses response_format='path' (TRTLLM_DISALLOW_LOCAL_MEDIA_PATH=1). "
    "Pass --response-format file (video) or --response-format url (image). "
    "There is no automatic fallback: it would silently "
    "change whether the media transfer is counted in the latency."
)


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


@dataclass
class VisualGenSampleRequest:
    """One dispatchable request with its merge already materialized."""

    prompt: str
    prompt_file: Optional[str] = None
    image_reference: Optional[str] = None
    video_reference: Optional[str] = None
    # A MediaReferenceItem object, or a list of them; None when there is no reference.
    image_reference_wire: Any = None
    video_reference_wire: Any = None
    params: VisualGenParams = field(default_factory=VisualGenParams)


@dataclass
class VisualGenWorkload:
    """Everything --workload resolves to."""

    backend: str
    requests: list[VisualGenSampleRequest]


@dataclass
class VisualGenRequestRecord:
    """One request's timings, resolved params and outcome.

    ``None`` means the timing is undefined for this backend (image has no poll
    phase) or was not reported; such samples are dropped from the aggregates.
    """

    index: int
    prompt: str
    params: dict[str, Any]
    prompt_file: Optional[str] = None
    image_reference: Optional[str] = None
    video_reference: Optional[str] = None
    success: bool = False
    start: float = 0.0
    end: float = 0.0
    client_e2e: Optional[float] = None
    client_gen: Optional[float] = None
    server_e2e: Optional[float] = None
    server_gen: Optional[float] = None
    server_denoise: Optional[float] = None
    poll_count: Optional[int] = None
    # Always a list: image backends can return n > 1, and a single shape lets a
    # consumer read the result without branching on the backend.
    output_paths: Optional[list[str]] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# --workload loader and merge
# --------------------------------------------------------------------------- #


def _sniff_workload_source(value: str) -> tuple[Any, Path]:
    """Resolve --workload to a parsed document and the base dir for relative paths.

    Leading ``[`` or ``{`` after strip() means inline content; no legal path
    starts with either. YAML is a JSON superset, so one parser covers both.
    """
    text = value.strip()
    if text.startswith("[") or text.startswith("{"):
        return yaml.safe_load(text), Path.cwd()
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(
            f"--workload {value!r} is not a file, and inline content must start with '[' or '{{'."
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f), path.parent


def _normalize_requests_doc(raw: Any) -> dict[str, Any]:
    if isinstance(raw, list):
        raw = {"requests": raw}
    if not isinstance(raw, dict):
        raise ValueError("--workload must be a mapping or a list of requests.")
    unknown = sorted(set(raw) - set(WORKLOAD_DOC_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown key(s) {unknown} in --workload; allowed: {list(WORKLOAD_DOC_KEYS)}."
        )
    requests = raw.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("A workload needs a non-empty 'requests' list.")
    return raw


def _resolve_scalar(name: str, cli_value: Optional[str], doc_value: Optional[str]) -> Optional[str]:
    """CLI supplies what the document omits; a disagreement is an error."""
    if cli_value is not None and doc_value is not None and cli_value != doc_value:
        raise ValueError(
            f"--{name} {cli_value!r} conflicts with '{name}: {doc_value}' in --workload."
        )
    return cli_value if cli_value is not None else doc_value


def _merge_extra_params(base: Optional[dict], override: Any) -> Optional[dict]:
    """Overlay ``extra_params`` per key, one level deep.

    No null filtering: the client cannot see ``extra_param_specs``, so dropping
    ``{"stg_sclae": null}`` here would turn a typo into a silent 200. Mirrors
    the server-side ``_merge_extra_params``.
    """
    if override is None:
        return None
    if not isinstance(override, dict):
        raise ValueError(f"extra_params must be a mapping, got {type(override).__name__}.")
    merged = dict(base or {})
    merged.update(override)
    return merged


def _merge_request(
    common: dict[str, Any], request_raw: dict[str, Any], index: int
) -> dict[str, Any]:
    """Overlay one raw request onto the common_params layer.

    ``request_raw`` keeps its nulls: stripping them here would let the common
    value win over a request that explicitly asked for the pipeline default.
    """
    present = sorted({"width", "height"} & set(request_raw))
    if len(present) == 1:
        raise ValueError(
            f"requests[{index}]: got {present[0]!r} without its pair; width and height must "
            "be set together, otherwise this request silently pairs with the common_params "
            "value for the other."
        )
    merged = {**common, **request_raw}
    if "extra_params" in request_raw:
        merged["extra_params"] = _merge_extra_params(
            common.get("extra_params"), request_raw["extra_params"]
        )
    return merged


def _resolve_reference(slot: str, reference: Any, base_dir: Path, index: int) -> tuple[str, Any]:
    """Return ``(label, wire)`` for one reference slot.

    A string is a local path: it is read and encoded here rather than at
    dispatch so a missing file fails before the run starts. A ``{content,
    format}`` object (or a list of them) is already in the wire form
    ``MediaReferenceItem`` declares and passes through untouched.
    """
    if isinstance(reference, (dict, list)):
        items = reference if isinstance(reference, list) else [reference]
        labels = []
        for item in items:
            if not isinstance(item, dict) or "content" not in item or "format" not in item:
                raise ValueError(
                    f"requests[{index}]: {slot} objects need 'content' and 'format' "
                    f"(path/url/base64); got {item!r}."
                )
            labels.append(item["content"] if item["format"] != "base64" else "<base64>")
        return ", ".join(labels), reference
    if not isinstance(reference, str):
        raise ValueError(
            f"requests[{index}]: {slot} must be a local path string or a "
            "{content, format} object."
        )
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    try:
        payload = path.read_bytes()
    except OSError as e:
        raise ValueError(f"requests[{index}]: cannot read {slot} {str(path)!r}: {e}") from e
    if not payload:
        raise ValueError(f"requests[{index}]: {slot} {str(path)!r} is empty.")
    encoded = base64.b64encode(payload).decode("ascii")
    return str(path.resolve()), {"content": encoded, "format": "base64"}


def _resize_requests(
    requests: list[VisualGenSampleRequest], total: int
) -> list[VisualGenSampleRequest]:
    """Cycle or truncate the expanded list to exactly ``total`` requests.

    Cycling repeats the list in order, so a mixed-shape workload keeps its
    proportions instead of over-weighting whichever request came first.
    """
    if total < 1:
        raise ValueError(f"--num-requests must be >= 1, got {total}.")
    if total <= len(requests):
        return requests[:total]
    out = list(requests)
    while len(out) < total:
        source = requests[len(out) % len(requests)]
        out.append(replace(source, params=source.params.model_copy(deep=True)))
    return out


def _resolve_prompt_file(reference: Any, base_dir: Path, index: int) -> tuple[str, str]:
    """Read a prompt file, in the three shapes Cosmos3's prompt files come in.

    A JSON object carrying ``prompt`` yields that field; one without it is a
    structured caption and goes out serialized, which is what the example does
    with the ``*_prompt.json`` a checkpoint ships; anything that is not JSON is
    plain text. Reading here rather than at dispatch means a missing or empty
    file fails before the run starts.
    """
    if not isinstance(reference, str):
        raise ValueError(f"requests[{index}]: prompt_file must be a path string.")
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"requests[{index}]: cannot read prompt_file {str(path)!r}: {e}") from e
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = raw.strip()
    if isinstance(payload, dict):
        prompt = payload["prompt"] if "prompt" in payload else json.dumps(payload)
    elif isinstance(payload, str):
        prompt = payload
    else:
        raise ValueError(
            f"requests[{index}]: prompt_file {str(path)!r} must hold a JSON object or "
            f"text, got {type(payload).__name__}."
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"requests[{index}]: prompt_file {str(path)!r} yields an empty prompt.")
    return str(path.resolve()), prompt


def _build_sample_request(
    merged: dict[str, Any], index: int, base_dir: Path
) -> VisualGenSampleRequest:
    merged = dict(merged)
    prompt = merged.pop("prompt", None)
    prompt_file = merged.pop("prompt_file", None)
    references = {slot: merged.pop(slot) for slot in REFERENCE_KEYS if slot in merged}
    if prompt_file is not None:
        if prompt is not None:
            raise ValueError(
                f"requests[{index}]: set 'prompt' or 'prompt_file', not both — which one "
                "the run measured would depend on a precedence rule rather than the document."
            )
        prompt_file, prompt = _resolve_prompt_file(prompt_file, base_dir, index)
    if not isinstance(prompt, str):
        raise ValueError(
            f"requests[{index}]: 'prompt' (or 'prompt_file') is required and must be a string."
        )
    try:
        params = VisualGenParams(**merged)
    except ValidationError as e:
        raise ValueError(f"requests[{index}]: invalid generation parameters:\n{e}") from e
    if (params.width is None) != (params.height is None):
        raise ValueError(
            f"requests[{index}]: resolved width={params.width!r} height={params.height!r}; the "
            "server rejects exactly one of them (HTTP 422). Set both, or neither."
        )
    resolved: dict[str, Any] = {}
    for slot, reference in references.items():
        if reference is None:
            continue
        label, wire = _resolve_reference(slot, reference, base_dir, index)
        resolved[slot], resolved[f"{slot}_wire"] = label, wire
    return VisualGenSampleRequest(prompt=prompt, prompt_file=prompt_file, params=params, **resolved)


def _document_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the document the CLI spells out.

    Each flag maps to a named part of the document -- fields to
    ``common_params``, ``--requests`` to the list -- so the CLI is a second way
    to write it, not a second way to run. Both spellings converge here, and the
    merge and every validation below have one implementation.
    """
    common: dict[str, Any] = {}
    for key in (*SCALAR_PARAM_FIELDS, *COMMON_INPUT_KEYS):
        value = getattr(args, key)
        if value is not None:
            common[key] = value
    if args.extra_params is not None:
        common["extra_params"] = _cli_json("--extra-params", args.extra_params)

    doc: dict[str, Any] = {"common_params": common}
    if args.requests is not None:
        doc["requests"] = _cli_json("--requests", args.requests)
    return doc


def _cli_json(flag: str, value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"{flag} is not JSON: {e}") from e


def load_workload(args: argparse.Namespace) -> VisualGenWorkload:
    """Resolve the workload from --workload, or from the request named on the CLI.

    Merging here rather than at dispatch is what makes a bad parameter fail
    before a multi-minute run starts, and lets every result record carry the
    params it was actually sent with.
    """
    named = sorted(key for key in CLI_KEYS if getattr(args, key) is not None)
    if args.workload and named:
        raise ValueError(
            f"--workload and {', '.join('--' + k.replace('_', '-') for k in named)} are "
            "alternatives: the document in a file, or the same document spelled on the "
            "CLI. Combining them would need a precedence rule between the two, which is "
            "what writing one document avoids."
        )
    if not args.workload and not named:
        raise ValueError("Pass --workload <document>, or spell one on the CLI.")

    if args.workload:
        doc, base_dir = _sniff_workload_source(args.workload)
    else:
        doc, base_dir = _document_from_args(args), Path.cwd()
    doc = _normalize_requests_doc(doc)

    backend = _resolve_scalar("backend", args.backend, doc.get("backend"))
    if backend is None:
        raise ValueError(
            "backend is required: set 'backend' in --workload, or pass --backend. There is "
            "no default, because it selects the route and so what the run measures: a "
            "checkpoint serving both modes answers the wrong one without complaining."
        )
    if backend not in BACKEND_ENDPOINTS:
        raise ValueError(f"backend must be one of {sorted(BACKEND_ENDPOINTS)}; got {backend!r}.")

    common_raw = dict(doc.get("common_params") or {})
    misplaced = sorted(set(REFERENCE_KEYS) & set(common_raw))
    if misplaced:
        raise ValueError(
            f"common_params: {', '.join(misplaced)} belongs to a request, not to every "
            "request. Move it into the 'requests' entry it conditions."
        )
    common_input = {key: common_raw.pop(key) for key in COMMON_INPUT_KEYS if key in common_raw}
    try:
        common_params = VisualGenParams(**common_raw)
    except ValidationError as e:
        raise ValueError(f"common_params: invalid generation parameters:\n{e}") from e
    common = {**common_input, **common_params.model_dump(exclude_unset=True)}

    requests: list[VisualGenSampleRequest] = []
    for index, raw in enumerate(doc["requests"]):
        if not isinstance(raw, dict):
            raise ValueError(f"requests[{index}] must be a mapping, got {type(raw).__name__}.")
        requests.append(_build_sample_request(_merge_request(common, raw, index), index, base_dir))

    if args.num_requests is not None:
        requests = _resize_requests(requests, args.num_requests)

    workload = VisualGenWorkload(
        backend=backend,
        requests=requests,
    )
    _validate_workload(workload)
    return workload


# --------------------------------------------------------------------------- #
# Pre-run validations
# --------------------------------------------------------------------------- #


def _validate_reference_backend(workload: VisualGenWorkload) -> None:
    """Reject the two reference/backend pairings that are a guaranteed 422.

    ``ImageGenerationRequest`` declares neither ``image`` nor
    ``image_reference`` and forbids extras; ``ImageEditRequest.image`` is
    required. Video is unconstrained: I2V/V2V ride the same route.
    """
    for index, request in enumerate(workload.requests):
        if request.image_reference is not None and workload.backend == "openai-images":
            raise ValueError(
                f"requests[{index}]: image_reference is not accepted by 'openai-images'; use "
                "'openai-image-edits' for image editing or 'openai-videos' for I2V/V2V."
            )
        if request.video_reference is not None and workload.backend != VIDEO_BACKEND:
            raise ValueError(
                f"requests[{index}]: video_reference is not accepted by {workload.backend!r}; "
                "only 'openai-videos' carries a video reference."
            )
        if workload.backend == "openai-image-edits":
            if request.image_reference is None:
                raise ValueError(
                    f"requests[{index}]: backend 'openai-image-edits' requires image_reference "
                    "(/v1/images/edits has a required 'image' field)."
                )
            wire = request.image_reference_wire
            if not (isinstance(wire, dict) and wire.get("format") == "base64"):
                raise ValueError(
                    f"requests[{index}]: 'openai-image-edits' takes a single base64 image, so "
                    "image_reference must be a local path or one {content, format: base64} "
                    "object; /v1/images/edits does not accept the video route's list form."
                )


def _validate_output_type(workload: VisualGenWorkload) -> None:
    """Reject an ``extra_params.output_type`` that contradicts the backend.

    Gate on the value, not the model: Cosmos3 uses image/video to select its
    mode table, while LTX-2 reuses the key for pt/pil, which carries no
    modality and must pass through untouched.
    """
    expected = MODALITY_BY_BACKEND[workload.backend]
    for index, request in enumerate(workload.requests):
        value = (request.params.extra_params or {}).get("output_type")
        if value in ("image", "video") and value != expected:
            raise ValueError(
                f"requests[{index}]: extra_params.output_type={value!r} contradicts backend "
                f"{workload.backend!r} (expects {expected!r})."
            )


def _validate_workload(workload: VisualGenWorkload) -> None:
    _validate_reference_backend(workload)
    _validate_output_type(workload)


# --------------------------------------------------------------------------- #
# Payload construction
# --------------------------------------------------------------------------- #


def _params_dump(params: VisualGenParams) -> dict[str, Any]:
    """Dump the merged params as sent.

    ``exclude_unset`` keeps a request's explicit ``null`` (the server treats an
    omitted field and an explicit null identically) while never inventing a
    default. ``image`` is a server-side artifact; the client carries the
    reference in the payload's own reference field.
    """
    dump = params.model_dump(exclude_unset=True)
    dump.pop("image", None)
    return dump


def build_payload(
    request: VisualGenSampleRequest,
    backend: str,
    model: str,
    response_format: str,
    output_format: Optional[str],
) -> dict[str, Any]:
    """Build the HTTP body for one request on one backend.

    A params dump is not a legal body: ``num_images_per_prompt`` is on no video
    wire model and ``num_frames`` / ``frame_rate`` on no image wire model, so
    ``extra="forbid"`` would 422 every request. The frame budget goes out as
    ``num_frames``; the wire's ``seconds`` alternative is derived server-side as
    ``int(seconds * frame_rate)``, which drops a frame at 25/30/50/60/120 fps.
    """
    payload: dict[str, Any] = {
        "prompt": request.prompt,
        "model": model,
        "response_format": response_format,
    }
    params = _params_dump(request.params)
    num_images = params.pop("num_images_per_prompt", None)

    if backend == VIDEO_BACKEND:
        if request.image_reference_wire is not None:
            # Typed field with a declared wire form -- the deprecated
            # ``input_reference`` sniffs the content to guess the modality.
            payload["image_reference"] = request.image_reference_wire
        if request.video_reference_wire is not None:
            payload["video_reference"] = request.video_reference_wire
        if output_format is not None:
            payload["format"] = output_format
    else:
        params.pop("num_frames", None)
        params.pop("frame_rate", None)
        if num_images is not None:
            payload["n"] = num_images
        if backend == "openai-image-edits":
            payload["image"] = request.image_reference_wire["content"]
            if output_format is not None:
                # ImageEditRequest's canonical name; ``format`` is only an alias.
                payload["output_format"] = output_format
        elif output_format is not None:
            payload["format"] = output_format

    payload.update(params)
    return payload


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #


def _get_headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'unused')}",
    }


def _parse_server_timing_header(headers: Any) -> dict[str, float]:
    """Parse required VisualGen Server-Timing metrics into seconds.

    Online VisualGen perf sanity gates on engine-side generation time, so a
    successful response without valid ``Server-Timing`` metadata is treated as
    a failed benchmark request instead of silently contributing a zero sample.
    """
    value = headers.get(SERVER_TIMING_HEADER)
    if value is None:
        raise ValueError(f"Missing VisualGen timing response header: {SERVER_TIMING_HEADER}")

    timings = {}
    for entry in value.split(","):
        parts = [part.strip() for part in entry.split(";")]
        name = parts[0]
        for parameter in parts[1:]:
            key, _, parameter_value = parameter.partition("=")
            if key.strip() == "dur":
                timings[name] = float(parameter_value) / 1000.0
                break
    return timings


def _get_server_timing_metric(
    timings: dict[str, float], name: str, *, require_positive: bool
) -> float:
    """Return a required Server-Timing metric, in seconds."""
    if name not in timings:
        raise ValueError(f"Missing VisualGen Server-Timing metric: {name}")
    timing = timings[name]
    if not math.isfinite(timing) or timing < 0 or (require_positive and timing <= 0):
        raise ValueError(f"Invalid VisualGen Server-Timing metric {name}: {timing}")
    return timing


def _record_server_timings(headers: Any, record: VisualGenRequestRecord) -> None:
    """Fill the three server-side timings from the Server-Timing header.

    ``total`` is optional -- the video route omits it when the job carries no
    arrival stamp -- so its absence leaves ``server_e2e`` unreported rather
    than failing the request.
    """
    timings = _parse_server_timing_header(headers)
    record.server_gen = _get_server_timing_metric(
        timings, VISUAL_GEN_GENERATION_TIMING, require_positive=True
    )
    record.server_denoise = _get_server_timing_metric(
        timings, VISUAL_GEN_DENOISE_TIMING, require_positive=False
    )
    record.server_e2e = timings.get(VISUAL_GEN_TOTAL_TIMING)


async def _dispatch_image(
    session: aiohttp.ClientSession,
    api_url: str,
    payload: dict[str, Any],
    record: VisualGenRequestRecord,
) -> None:
    """POST /v1/images/{generations,edits}: one leg, result in the response body."""
    async with session.post(url=api_url, json=payload, headers=_get_headers()) as response:
        if response.status != 200:
            record.error = f"HTTP {response.status}: {await response.text()}"
            return
        body = await response.json()
        _record_server_timings(response.headers, record)

    if payload["response_format"] == "path":
        record.output_paths = [item["path"] for item in body["data"]]
    record.success = True


async def _dispatch_video(
    session: aiohttp.ClientSession,
    api_url: str,
    payload: dict[str, Any],
    record: VisualGenRequestRecord,
    t0: float,
    poll_interval: float,
    request_timeout: float,
) -> None:
    """POST /v1/videos -> poll status -> GET /content.

    ``/content`` is fetched even for ``response_format='path'``: the status GET
    carries no Server-Timing (``VideoJob.timing_metrics`` is excluded from the
    wire), so it is the only source of the three server-side timings.
    """
    async with session.post(url=api_url, json=payload, headers=_get_headers()) as response:
        body = await response.text()
        if response.status != 202:
            record.error = f"HTTP {response.status}: {body}"
            record.poll_count = 0
            return
        video_id = json.loads(body)["id"]

    status_url = f"{api_url}/{video_id}"
    record.poll_count = 0
    # aiohttp's timeout bounds a single call, so the poll loop needs its own
    # deadline: a crashed worker leaves the job at "generating" forever.
    deadline = t0 + request_timeout
    # The server flips to "postprocessing" once inference is done and before it
    # encodes and saves, so gen_latency ends there rather than folding the encode
    # in. A job that finishes between two polls is only ever seen as "completed";
    # stamping on either status keeps that case measured instead of null.
    generated_at = None
    while True:
        async with session.get(url=status_url, headers=_get_headers()) as response:
            if response.status != 200:
                record.error = f"HTTP {response.status}: {await response.text()}"
                return
            job = await response.json()
        record.poll_count += 1
        status = job.get("status")
        if generated_at is None and status in ("postprocessing", "completed"):
            generated_at = time.perf_counter() - t0
        if status == "completed":
            break
        if status == "failed":
            record.client_gen = time.perf_counter() - t0
            record.error = f"Video job {video_id} failed: {job.get('error')}"
            return
        if time.perf_counter() > deadline:
            record.error = (
                f"Video job {video_id} did not reach a terminal status within "
                f"{request_timeout}s (last status: {status})"
            )
            return
        await asyncio.sleep(poll_interval)
    record.client_gen = generated_at

    async with session.get(url=f"{status_url}/content", headers=_get_headers()) as response:
        if response.status != 200:
            record.error = f"HTTP {response.status}: {await response.text()}"
            return
        if payload["response_format"] == "path":
            record.output_paths = [(await response.json())["output_path"]]
        else:
            await response.read()
        _record_server_timings(response.headers, record)
    record.success = True


async def dispatch_request(
    session: aiohttp.ClientSession,
    backend: str,
    api_url: str,
    payload: dict[str, Any],
    record: VisualGenRequestRecord,
    benchmark_start: float,
    poll_interval: Optional[float],
    request_timeout: float,
    pbar: Optional[tqdm] = None,
) -> VisualGenRequestRecord:
    """Run one request end to end; client_e2e is recorded even on failure."""
    t0 = time.perf_counter()
    record.start = t0 - benchmark_start
    try:
        if backend == VIDEO_BACKEND:
            await _dispatch_video(
                session, api_url, payload, record, t0, poll_interval, request_timeout
            )
        else:
            await _dispatch_image(session, api_url, payload, record)
    except Exception:
        record.success = False
        record.error = "".join(traceback.format_exception(*sys.exc_info()))
    finally:
        record.end = time.perf_counter() - benchmark_start
        record.client_e2e = record.end - record.start
        if pbar:
            pbar.update(1)
    return record


# Carried in the payload but not generation parameters; references are recorded
# separately, by locator rather than by their base64 payload.
_NON_PARAM_PAYLOAD_KEYS = frozenset({"prompt", "model", "image", *REFERENCE_KEYS})


def _make_record(
    index: int, request: VisualGenSampleRequest, payload: dict[str, Any]
) -> VisualGenRequestRecord:
    """Record the parameters as sent -- a backend drops the ones it cannot carry."""
    return VisualGenRequestRecord(
        index=index,
        prompt=request.prompt,
        prompt_file=request.prompt_file,
        params={k: v for k, v in payload.items() if k not in _NON_PARAM_PAYLOAD_KEYS},
        image_reference=request.image_reference,
        video_reference=request.video_reference,
    )


async def benchmark(
    *,
    backend: str,
    base_url: str,
    model: str,
    workload: VisualGenWorkload,
    response_format: str,
    output_format: Optional[str],
    disable_tqdm: bool,
    max_concurrency: Optional[int],
    no_test_input: bool,
    request_timeout: float,
    poll_interval: Optional[float],
) -> tuple[list[VisualGenRequestRecord], float]:
    api_url = f"{base_url}{BACKEND_ENDPOINTS[backend]}"
    payloads = [
        build_payload(request, backend, model, response_format, output_format)
        for request in workload.requests
    ]

    pbar = None if disable_tqdm else tqdm(total=len(payloads), desc="Benchmarking")
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_dispatch(*args, **kwargs) -> VisualGenRequestRecord:
        if semaphore is None:
            return await dispatch_request(*args, **kwargs)
        async with semaphore:
            return await dispatch_request(*args, **kwargs)

    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(
        trust_env=True,
        timeout=timeout,
        connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True),
    ) as session:
        if not no_test_input:
            print("Starting initial single prompt test run...")
            test_record = await dispatch_request(
                session,
                backend,
                api_url,
                payloads[0],
                _make_record(0, workload.requests[0], payloads[0]),
                time.perf_counter(),
                poll_interval,
                request_timeout,
            )
            if not test_record.success:
                message = (
                    "Initial test run failed - Please make sure benchmark arguments are "
                    f"correctly specified. Error: {test_record.error}"
                )
                if "TRTLLM_DISALLOW_LOCAL_MEDIA_PATH" in (test_record.error or ""):
                    message = f"{message}\n{PATH_DISABLED_HINT}"
                raise ValueError(message)
            print("Initial test run completed. Starting main benchmark run...")
        else:
            print("Skipping initial test run. Starting main benchmark run...")

        print(f"Maximum request concurrency: {max_concurrency}")

        benchmark_start = time.perf_counter()
        tasks = [
            asyncio.create_task(
                limited_dispatch(
                    session,
                    backend,
                    api_url,
                    payload,
                    _make_record(index, request, payload),
                    benchmark_start,
                    poll_interval,
                    request_timeout,
                    pbar,
                )
            )
            for index, (request, payload) in enumerate(zip(workload.requests, payloads))
        ]
        records: list[VisualGenRequestRecord] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    return records, time.perf_counter() - benchmark_start


# --------------------------------------------------------------------------- #
# Aggregation and output
# --------------------------------------------------------------------------- #


def _percentile_key(percentile: float) -> str:
    return f"p{int(percentile) if int(percentile) == percentile else percentile}"


def _samples(records: list[VisualGenRequestRecord], name: str) -> list[float]:
    return [getattr(r, name) for r in records if r.success and getattr(r, name) is not None]


def _stats(samples: list[float], selected_percentiles: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            **{column: 0.0 for column in STAT_COLUMNS},
            "percentiles": {_percentile_key(p): 0.0 for p in selected_percentiles},
        }
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "percentiles": {
            _percentile_key(p): float(np.percentile(samples, p)) for p in selected_percentiles
        },
    }


def _output_rate(
    records: list[VisualGenRequestRecord], backend: str, duration: float
) -> tuple[str, Optional[float]]:
    """Produced-output rate.

    ``n`` is a batch dimension (one batched forward), so it counts toward the
    output and never divides a latency. ``None`` when a request left
    ``num_frames`` to the pipeline and the client cannot know the frame count.
    """
    done = [record for record in records if record.success]
    if backend == VIDEO_BACKEND:
        key = "frames_per_second"
        frames = [record.params.get("num_frames") for record in done]
        count = None if any(n is None for n in frames) else sum(frames)
    else:
        key = "images_per_second"
        count = sum(int(record.params.get("num_images_per_prompt", 1)) for record in done)
    if count is None:
        return key, None
    return key, count / duration if duration > 0 else 0.0


def _record_json(record: VisualGenRequestRecord, backend: str) -> dict[str, Any]:
    data: dict[str, Any] = {
        "index": record.index,
        "success": record.success,
        "prompt": record.prompt,
        "params": record.params,
        "start": record.start,
        "end": record.end,
        "client_e2e": record.client_e2e,
        "client_gen": record.client_gen,
        "server_e2e": record.server_e2e,
        "server_gen": record.server_gen,
        "server_denoise": record.server_denoise,
        "poll_count": record.poll_count,
    }
    if record.prompt_file is not None:
        data["prompt_file"] = record.prompt_file
    for slot in REFERENCE_KEYS:
        value = getattr(record, slot)
        if value is not None:
            data[slot] = value
    data["output_paths"] = record.output_paths
    data["error"] = record.error
    return data


def build_visual_gen_result(
    *,
    backend: str,
    model: str,
    duration: float,
    records: list[VisualGenRequestRecord],
    selected_percentiles: list[float],
    config: dict[str, Any],
    save_detailed: bool,
) -> dict[str, Any]:
    """Assemble the result JSON; stdout is printed from this same dict.

    Per-request arrays and the server-side timings exist only under
    ``save_detailed``.
    """
    completed = sum(1 for record in records if record.success)
    rate_key, rate = _output_rate(records, backend, duration)

    result: dict[str, Any] = {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "backend": backend,
        "model": model,
        "duration": duration,
        "config": config,
        "total_requests": len(records),
        "completed": completed,
        "request_throughput": completed / duration if duration > 0 else 0.0,
        rate_key: rate,
        "e2e_latency": _stats(_samples(records, "client_e2e"), selected_percentiles),
    }
    if backend == VIDEO_BACKEND:
        result["gen_latency"] = _stats(_samples(records, "client_gen"), selected_percentiles)
    if save_detailed:
        result["timings"] = {
            name: _stats(_samples(records, name), selected_percentiles)
            for name in SERVER_TIMING_FIELDS
        }
        result["requests"] = [_record_json(record, backend) for record in records]
    return result


def print_visual_gen_results(result: dict[str, Any], selected_percentiles: list[float]) -> None:
    is_video = result["backend"] == VIDEO_BACKEND
    failed = result["total_requests"] - result["completed"]

    print("{s:{c}^{n}}".format(s=" Benchmark Result (VisualGen) ", n=TABLE_WIDTH, c="="))
    print("{:<32} {}".format("Backend:", result["backend"]))
    print("{:<32} {}".format("Model:", result["model"]))
    print(
        "{:<32} {} / {} / {}".format(
            "Total / Successful / Failed:",
            result["total_requests"],
            result["completed"],
            failed,
        )
    )
    print("{:<32} {:.2f}".format("Benchmark duration (s):", result["duration"]))
    print("{:<32} {:.4f}".format("Request throughput (req/s):", result["request_throughput"]))
    rate_label = "Frames per second:" if is_video else "Images per second:"
    rate = result["frames_per_second"] if is_video else result["images_per_second"]
    print("{:<32} {}".format(rate_label, "n/a" if rate is None else f"{rate:.2f}"))
    print("{:<32} {}".format("Max concurrency:", result["config"]["max_concurrency"]))

    if failed:
        print("=" * TABLE_WIDTH)
        print(f"  !!! {failed} FAILED REQUESTS - CHECK LOG FOR ERRORS !!!")
        print("=" * TABLE_WIDTH)

    percentile_keys = [_percentile_key(p) for p in selected_percentiles]
    print("-" * TABLE_WIDTH)
    print(
        "{:<14}".format("Timing (s)")
        + "".join(f"{column:>9}" for column in (*STAT_COLUMNS, *percentile_keys))
    )
    for name in ("e2e_latency", "gen_latency"):
        block = result.get(name)
        if block is None:
            continue
        values = [block[column] for column in STAT_COLUMNS]
        values += [block["percentiles"][key] for key in percentile_keys]
        print("{:<14}".format(name) + "".join(f"{value:>9.3f}" for value in values))
    print("=" * TABLE_WIDTH)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


async def fetch_served_model(base_url: str) -> str:
    """Return ``data[0].id`` from GET /v1/models."""
    probe_timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(trust_env=True, timeout=probe_timeout) as session:
        async with session.get(f"{base_url}/v1/models", headers=_get_headers()) as response:
            if response.status != 200:
                raise ValueError(
                    f"GET {base_url}/v1/models returned HTTP {response.status}. Is trtllm-serve up?"
                )
            data = (await response.json()).get("data") or []
    if not data:
        raise ValueError(f"GET {base_url}/v1/models returned an empty model list.")
    return data[0]["id"]


def _same_model(candidate: str, served: str) -> bool:
    """Compare basenames: the server reports ``Path(model).name`` for a directory."""
    return os.path.basename(candidate.rstrip("/")) == os.path.basename(served.rstrip("/"))


def resolve_model(args: argparse.Namespace, served: str) -> str:
    """Resolve the label and wire ``model`` field: --model, else the served id."""
    if args.model is None:
        return served
    if not _same_model(args.model, served):
        raise ValueError(f"--model {args.model!r} does not match the served model {served!r}.")
    return args.model


def resolve_poll_interval(args: argparse.Namespace, backend: str) -> Optional[float]:
    """``None`` off the video route, where nothing polls."""
    return args.poll_interval if backend == VIDEO_BACKEND else None


def build_arg_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(
        description="Benchmark VisualGen (image/video generation) serving."
    )

    conn_group = parser.add_argument_group("Connection")
    conn_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id or checkpoint path sent as the request 'model' field and used to "
        "label results. Default: the id reported by GET /v1/models.",
    )
    conn_group.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=list(BACKEND_ENDPOINTS),
        help="Backend API type. Supplies the 'backend' key when --workload omits it; "
        "conflicting with it is an error. Required from one of the two.",
    )
    conn_group.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
    conn_group.add_argument("--port", type=int, default=8000, help="Server port.")

    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Resize the workload to exactly this many requests, cycling the list "
        "in order or truncating it. Default: send the document as written.",
    )
    input_group.add_argument(
        "--workload",
        type=str,
        default=None,
        help="Workload document: a YAML/JSON file path, or inline content starting with "
        "'{' (full mapping) or '[' (bare requests list) -- a path starting with either "
        "character is not addressable. Top-level keys: backend, common_params, "
        "requests. Alternative to spelling the same document out below.",
    )

    # One flag per document key, same name, so a command line and a file
    # describe a workload identically. Either spelling, never both.
    request_group = parser.add_argument_group(
        "Workload on the CLI",
        "The same document, spelled out: these fields are its common_params, and "
        "--requests is its requests list, which it needs just as a file does. "
        "Alternative to --workload.",
    )
    request_group.add_argument(
        "--requests",
        type=str,
        default=None,
        help="The requests list as JSON, in the document's own form, e.g. "
        '\'[{"prompt": "a fox"}, {"prompt": "a cat", "seed": 7}]\'. Each entry '
        "overrides the fields below per key. Required, as it is in a file.",
    )
    for name, kind in SCALAR_PARAM_FIELDS.items():
        request_group.add_argument(
            f"--{name.replace('_', '-')}",
            type=kind,
            default=None,
            help=VisualGenParams.model_fields[name].description,
        )
    for name, text in zip(COMMON_INPUT_KEYS, ("The prompt text.", "Path to a prompt file.")):
        request_group.add_argument(f"--{name.replace('_', '-')}", type=str, default=None, help=text)
    request_group.add_argument(
        "--extra-params", type=str, default=None, help="Per-pipeline parameters, as a JSON object."
    )

    traffic_group = parser.add_argument_group("Traffic Control")
    traffic_group.add_argument(
        "--max-concurrency", type=int, default=None, help="Maximum concurrent requests."
    )
    traffic_group.add_argument(
        "--request-timeout",
        type=float,
        default=6 * 60 * 60,
        help="Request timeout in seconds (default: 6 hours).",
    )
    traffic_group.add_argument(
        "--response-format",
        type=str,
        default="path",
        help="How the server returns media (default: %(default)s). Run-level: mixing "
        "transport modes within one run makes the aggregate latency incomparable.",
    )
    traffic_group.add_argument(
        "--format",
        type=str,
        default=None,
        help="Encoding the server writes: mp4/avi/auto for video, png/webp/jpeg for "
        "images. Default: the server's, which for video is 'auto' -- without ffmpeg "
        "that is AVI/MJPEG, a different encode inside the measured window.",
    )
    traffic_group.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help=f"Job status poll interval in seconds for {VIDEO_BACKEND} "
        "(default: %(default)s). Image backends are synchronous and ignore it.",
    )
    traffic_group.add_argument(
        "--no-test-input", action="store_true", help="Skip the initial single-prompt test run."
    )
    traffic_group.add_argument("--disable-tqdm", action="store_true", help="Disable progress bar.")

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--save-result", action="store_true", help="Save results to JSON file."
    )
    output_group.add_argument(
        "--save-detailed", action="store_true", help="Include per-request details in saved results."
    )
    output_group.add_argument(
        "--result-dir", type=str, default=None, help="Directory for result files."
    )
    output_group.add_argument(
        "--result-filename", type=str, default=None, help="Custom result filename."
    )
    output_group.add_argument(
        "--metric-percentiles",
        type=str,
        default="50,90,99",
        help="Comma-separated percentile values (default: '50,90,99').",
    )
    output_group.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        type=str,
        nargs="*",
        default=None,
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )

    return parser


def main(args: argparse.Namespace):
    print(args)

    base_url = f"http://{args.host}:{args.port}"
    workload = load_workload(args)
    model = resolve_model(args, asyncio.run(fetch_served_model(base_url)))
    poll_interval = resolve_poll_interval(args, workload.backend)
    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]

    if not args.save_detailed:
        _warn(
            "--save-detailed is off: the result JSON keeps only run-level averages, and a "
            "heterogeneous run cannot be attributed without the per-request records."
        )

    gc.disable()

    records, duration = asyncio.run(
        benchmark(
            backend=workload.backend,
            base_url=base_url,
            model=model,
            workload=workload,
            response_format=args.response_format,
            output_format=args.format,
            disable_tqdm=args.disable_tqdm,
            max_concurrency=args.max_concurrency,
            no_test_input=args.no_test_input,
            request_timeout=args.request_timeout,
            poll_interval=poll_interval,
        )
    )

    for record in records:
        if not record.success:
            _warn(f"request {record.index} failed: {record.error}")
    if not any(record.success for record in records):
        _warn(
            "All requests failed. This is likely due to a misconfiguration on the "
            "benchmark arguments."
        )

    config: dict[str, Any] = {
        "num_requests": len(records),
        "max_concurrency": args.max_concurrency,
        "response_format": args.response_format,
        "format": args.format,
    }
    if workload.backend == VIDEO_BACKEND:
        config["poll_interval"] = poll_interval

    result = build_visual_gen_result(
        backend=workload.backend,
        model=model,
        duration=duration,
        records=records,
        selected_percentiles=selected_percentiles,
        config=config,
        save_detailed=args.save_detailed,
    )
    print_visual_gen_results(result, selected_percentiles)

    if args.save_result:
        for item in args.metadata or []:
            if "=" not in item:
                raise ValueError("Invalid metadata format. Please use KEY=VALUE format.")
            key, value = item.split("=", 1)
            result[key.strip()] = value.strip()

        base_model = model.rstrip("/").split("/")[-1]
        max_concurrency_str = (
            f"-concurrency{args.max_concurrency}" if args.max_concurrency is not None else ""
        )
        file_name = args.result_filename or (
            f"{workload.backend}{max_concurrency_str}-{base_model}-{result['date']}.json"
        )
        if args.result_dir:
            os.makedirs(args.result_dir, exist_ok=True)
            file_name = os.path.join(args.result_dir, file_name)

        with open(file_name, "w", encoding="utf-8") as outfile:
            json.dump(result, outfile, indent=2)

        print(f"Results saved to: {file_name}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
