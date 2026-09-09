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
``BENCHMARKING_VISUAL_GEN.md``, beside this file.
"""

import argparse
import asyncio
import base64
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from argparse import ArgumentParser as FlexibleArgumentParser
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional, Union, get_args

import aiohttp
import numpy as np
import yaml
from pydantic import Field, PrivateAttr, ValidationError, create_model, model_validator
from tqdm.asyncio import tqdm

from tensorrt_llm.llmapi.utils import StrictBaseModel
from tensorrt_llm.serve.openai_protocol import (
    ImageEditRequest,
    ImageGenerationRequest,
    MediaReferenceItem,
    VideoGenerationRequest,
)
from tensorrt_llm.serve.visual_gen_metrics import (
    SERVER_TIMING_HEADER,
    VISUAL_GEN_DENOISE_TIMING,
    VISUAL_GEN_GENERATION_TIMING,
    VISUAL_GEN_POST_DENOISE_TIMING,
    VISUAL_GEN_PRE_DENOISE_TIMING,
    VISUAL_GEN_TOTAL_TIMING,
)
from tensorrt_llm.visual_gen.params import VisualGenParams

# 2 is the workload-document shape. The shape before it carried no version, so a
# result without the key reads as 1.
RESULT_SCHEMA_VERSION = 2

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
# The request model each route validates against.
WIRE_MODEL = {
    "openai-images": ImageGenerationRequest,
    "openai-image-edits": ImageEditRequest,
    VIDEO_BACKEND: VideoGenerationRequest,
}
# Where the OpenAI-compatible wire spells a VisualGenParams field differently.
# The document keeps the API's name; only the payload uses these.
WIRE_ALIAS = {"num_images_per_prompt": "n", "image_reference": "image"}

# Reference slots the loader resolves itself, so a document can name a local file
# relative to itself and an unreadable one fails before the run.
# Read off the params rather than listed, so a slot the API gains is addressable.
REFERENCE_KEYS = tuple(name for name in VisualGenParams.model_fields if name.endswith("_reference"))
# What one generation is given. common_params carries generation parameters, so
# none of these belong there: spread over every request they would describe a run
# nobody asked for.
REQUEST_INPUT_KEYS = ("prompt", "prompt_file", *REFERENCE_KEYS)


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
CLI_KEYS = (*SCALAR_PARAM_FIELDS, "extra_params", "requests")

STAT_FUNCS = {"mean": np.mean, "median": np.median, "std": np.std, "min": np.min, "max": np.max}
STAT_COLUMNS = tuple(STAT_FUNCS)

PATH_DISABLED_HINT = (
    "The server refuses response_format='path' (TRTLLM_DISALLOW_LOCAL_MEDIA_PATH=1). "
    "Pass --response-format file (video) or --response-format url (image). "
    "There is no automatic fallback: it would silently "
    "change whether the media transfer is counted in the latency."
)


def _reject_request_only_key(cls, data: Any) -> Any:
    """``extra_forbidden`` would say it is not allowed, not where it belongs."""
    if isinstance(data, dict):
        misplaced = sorted(key for key in data if key in REQUEST_INPUT_KEYS)
        if misplaced:
            raise ValueError(
                f"{', '.join(misplaced)} belongs to a request, not to every request. "
                "Move it into the 'requests' entry it applies to."
            )
    return data


class VisualGenBenchRequest(StrictBaseModel):
    """One entry of ``requests``, in the fields its route accepts.

    Resolution replaces what the document named with what a request carries --
    a prompt file with its text, a relative reference path with its absolute
    one -- so the locator worth recording is kept here.
    """

    _original: dict[str, str] = PrivateAttr(default_factory=dict)


class VisualGenBenchWorkload(StrictBaseModel):
    """The --workload document, and what the CLI spells out.

    Each route subclasses this with the fields that route accepts, so a field
    the route cannot carry fails at load instead of going out unnoticed.
    """

    backend: str
    common_params: StrictBaseModel
    requests: list[VisualGenBenchRequest] = Field(min_length=1)


def _carried(backend: str, field_name: str) -> bool:
    """Whether this route's request model has a slot for the field."""
    wire = WIRE_MODEL[backend].model_fields
    return field_name in wire or WIRE_ALIAS.get(field_name, field_name) in wire


def _document_model(
    name: str,
    backend: str,
    extra: dict[str, Any],
    base: type[StrictBaseModel] = StrictBaseModel,
    **kwargs: Any,
) -> type[StrictBaseModel]:
    """A document layer for one route, carrying the fields that route accepts.

    Derived rather than declared so the document cannot name a field the API
    does not have, and a new parameter is expressible without an edit here.
    Every reference slot is dropped: a request adds back the ones its route
    carries, and ``common_params`` gets none, so the schema itself says a
    reference conditions one generation rather than every one.
    """
    fields: dict[str, Any] = {
        field_name: (spec.annotation, spec)
        for field_name, spec in VisualGenParams.model_fields.items()
        if not field_name.endswith("_reference") and _carried(backend, field_name)
    }
    fields.update(extra)
    return create_model(name, __base__=base, **kwargs, **fields)


_REQUEST_INPUT_FIELDS: dict[str, Any] = {
    key: (Optional[str], None) for key in REQUEST_INPUT_KEYS if key not in REFERENCE_KEYS
}


class VisualGenBenchMediaRef(MediaReferenceItem):
    """One media reference in a workload document; ``format`` defaults to ``path``."""

    format: Literal["path", "url", "base64"] = "path"


def _reference_fields(backend: str) -> dict[str, Any]:
    """The reference slots this route carries.

    One reference or a list of them, in the wire form ``MediaReferenceItem``
    declares. ``/v1/images/edits`` has a required ``image``, so its slot is
    required here rather than checked later.
    """
    one_or_more = Union[VisualGenBenchMediaRef, list[VisualGenBenchMediaRef]]
    required = backend == "openai-image-edits"
    return {
        slot: (one_or_more, ...) if required else (Optional[one_or_more], None)
        for slot in REFERENCE_KEYS
        if _carried(backend, slot)
    }


def _workload_model(backend: str) -> type[VisualGenBenchWorkload]:
    """The document one route accepts, in three derived layers."""
    prefix = backend.title().replace("-", "")
    common = _document_model(
        f"{prefix}Common",
        backend,
        {},
        __validators__={
            "_reject_request_only_key": model_validator(mode="before")(
                classmethod(_reject_request_only_key)
            )
        },
    )
    request = _document_model(
        f"{prefix}Request",
        backend,
        {**_REQUEST_INPUT_FIELDS, **_reference_fields(backend)},
        base=VisualGenBenchRequest,
    )
    return create_model(
        f"{prefix}Workload",
        __base__=VisualGenBenchWorkload,
        backend=(Literal[backend], ...),
        common_params=(common, Field(default_factory=common)),
        requests=(list[request], Field(min_length=1)),
    )


WORKLOAD_MODEL = {backend: _workload_model(backend) for backend in WIRE_MODEL}


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


class VisualGenRequestRecord(StrictBaseModel):
    """One request's timings, resolved params and outcome.

    One of these per request is written to the result JSON under ``requests``
    by ``--save-detailed``, and the run's aggregate series are these fields
    over the run. ``None`` means the timing is undefined for this backend
    (image has no poll phase) or was not reported; such samples are dropped
    from the aggregates rather than counted as zero.
    """

    index: int = Field(description="Position of this request in the workload.")
    prompt: str = Field(description="The prompt text sent.")
    params: dict[str, Any] = Field(
        description="The merged parameters as sent, which is what this request measured."
    )
    prompt_file: Optional[str] = Field(
        default=None,
        description="The prompt file the document named, by locator. Resolution replaces "
        "it with its text, so the locator is what is worth recording.",
    )
    action_file: Optional[str] = Field(
        default=None, description="The action trajectory the document named, by locator."
    )
    image_reference: Optional[str] = Field(
        default=None,
        description="The image reference the document named, by locator -- a path, a URL, "
        "or '<base64>'. The bytes themselves would be megabytes wide and useless here.",
    )
    video_reference: Optional[str] = Field(
        default=None, description="The video reference the document named, by locator."
    )
    audio_reference: Optional[str] = Field(
        default=None, description="The audio reference the document named, by locator."
    )
    success: bool = Field(
        default=False,
        description="Whether the request completed. A run with any failure is not a result.",
    )
    start: float = Field(default=0.0, description="Seconds from the run's start to this send.")
    end: float = Field(default=0.0, description="Seconds from the run's start to this finish.")
    client_e2e: Optional[float] = Field(
        default=None,
        description="Seconds from sending the request until its result has been fully read "
        "from the server. Aggregated as e2e_latency.",
    )
    client_gen: Optional[float] = Field(
        default=None,
        description="Seconds from sending the request until the job first reports "
        "'postprocessing' or 'completed'. Video only, and its granularity is "
        "--poll-interval. Aggregated as gen_latency.",
    )
    server_e2e: Optional[float] = Field(
        default=None,
        description="Server-measured seconds from request arrival to the finished artifact: "
        "the encoded file on the video backend, the encoded image in the body on the image "
        "backends.",
    )
    server_gen: Optional[float] = Field(
        default=None,
        description="Server-measured seconds in the engine's inference call, what "
        "VisualGen.generate() costs, before any encoding or persistence. It excludes "
        "network and poll granularity, which makes it the metric to watch for regressions.",
    )
    server_pre_denoise: Optional[float] = Field(
        default=None,
        description="Server-measured seconds of text encoding, latent prep and conditioning, "
        "on the GPU stream.",
    )
    server_denoise: Optional[float] = Field(
        default=None, description="Server-measured seconds of this request's whole denoise loop."
    )
    server_post_denoise: Optional[float] = Field(
        default=None,
        description="Server-measured seconds of VAE decode, format conversion and audio "
        "decode, on the GPU stream.",
    )
    poll_count: Optional[int] = Field(
        default=None, description="Status polls this request took. Video only."
    )
    output_paths: Optional[list[str]] = Field(
        default=None,
        description="Where this request's media landed: server-side paths under "
        "--response-format path, local files under --output-media-dir. Always a list, since "
        "an image request with n > 1 has several and a single shape lets a consumer read "
        "the result without branching on the backend.",
    )
    error: Optional[str] = Field(default=None, description="Why the request failed, if it did.")


# Aggregated as their own series, so a timing added to the record above is
# reported without a second list to keep in step with it.
SERVER_TIMING_FIELDS = tuple(
    name for name in VisualGenRequestRecord.model_fields if name.startswith("server_")
)


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
    common: dict[str, Any], request: VisualGenBenchRequest, index: int
) -> dict[str, Any]:
    """Overlay one request onto the common_params layer.

    ``exclude_unset`` keeps a request's explicit ``null`` while never inventing
    a default, so a request can send a field back to the pipeline's own value
    even when common_params names it.
    """
    request_raw = request.model_dump(exclude_unset=True)
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
    """Return ``(label, resolved)`` for one reference slot.

    A ``path`` is made absolute against the document's directory and opened
    here, so an unreadable one fails before the run starts; the server does the
    reading, so the file never crosses the wire. ``url`` and ``base64`` are
    already what goes out.
    """
    items = reference if isinstance(reference, list) else [reference]
    labels: list[str] = []
    resolved: list[VisualGenBenchMediaRef] = []
    for item in items:
        if item.format != "path":
            labels.append("<base64>" if item.format == "base64" else item.content)
            resolved.append(item)
            continue
        path = Path(item.content).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        try:
            with path.open("rb") as handle:
                empty = not handle.read(1)
        except OSError as e:
            raise ValueError(f"requests[{index}]: cannot read {slot} {str(path)!r}: {e}") from e
        if empty:
            raise ValueError(f"requests[{index}]: {slot} {str(path)!r} is empty.")
        labels.append(str(path.resolve()))
        resolved.append(item.model_copy(update={"content": str(path.resolve())}))
    return ", ".join(labels), resolved if isinstance(reference, list) else resolved[0]


def _resize_requests(
    requests: list[VisualGenBenchRequest], total: int
) -> list[VisualGenBenchRequest]:
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
        out.append(requests[len(out) % len(requests)].model_copy(deep=True))
    return out


def _resolve_prompt_file(reference: Any, base_dir: Path, index: int) -> tuple[str, str]:
    """Read a prompt file, in the shapes Cosmos3's prompt files come in.

    A JSON object carrying ``prompt`` yields that field, serialized when it is
    itself an object: the action prompts hold the trained caption there, as
    nested ``cinematography`` / ``actions`` / canvas keys. An object without a
    ``prompt`` key is that caption on its own and goes out serialized; anything
    that is not JSON is plain text. Reading here rather than at dispatch means a
    missing or empty file fails before the run starts.
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
        prompt = payload["prompt"] if "prompt" in payload else payload
        if isinstance(prompt, dict):
            prompt = json.dumps(prompt)
    elif isinstance(payload, str):
        prompt = payload
    else:
        raise ValueError(
            f"requests[{index}]: prompt_file {str(path)!r} must hold a JSON object or "
            f"text, got {type(payload).__name__}."
        )
    if not isinstance(prompt, str):
        raise ValueError(
            f"requests[{index}]: prompt_file {str(path)!r} carries a "
            f"{type(prompt).__name__} 'prompt'; it must be text or a JSON object."
        )
    if not prompt.strip():
        raise ValueError(f"requests[{index}]: prompt_file {str(path)!r} yields an empty prompt.")
    return str(path.resolve()), prompt


def _resolve_action_file(reference: Any, base_dir: Path, index: int) -> tuple[str, list]:
    """Read a ``[T, D]`` action trajectory, the shape ``--action_json`` takes.

    Resolved into ``extra_params.action`` and dropped, since the server accepts
    only the keys the pipeline declares. Reading here keeps a few hundred
    literals out of the document and fails a malformed one before the run.
    """
    if not isinstance(reference, str):
        raise ValueError(f"requests[{index}]: extra_params.action_file must be a path string.")
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"requests[{index}]: cannot read action_file {str(path)!r}: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"requests[{index}]: action_file {str(path)!r} is not JSON: {e}") from e
    rows = payload if isinstance(payload, list) else []
    rectangular = (
        bool(rows)
        and all(isinstance(row, list) and row for row in rows)
        and len({len(row) for row in rows}) == 1
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for row in rows
            for value in row
        )
    )
    if not rectangular:
        raise ValueError(
            f"requests[{index}]: action_file {str(path)!r} must hold a non-empty [T, D] list "
            f"of equal-length numeric rows."
        )
    return str(path.resolve()), rows


def _resolve_request(
    merged: dict[str, Any], index: int, base_dir: Path, model: type[VisualGenBenchRequest]
) -> VisualGenBenchRequest:
    """Turn one merged request into the dispatchable form.

    Files are read here rather than at dispatch, so a missing one fails before
    a multi-minute run instead of part-way through it.
    """
    merged = dict(merged)
    prompt_file = merged.pop("prompt_file", None)
    if prompt_file is not None:
        if merged.get("prompt") is not None:
            raise ValueError(
                f"requests[{index}]: set 'prompt' or 'prompt_file', not both — which one "
                "the run measured would depend on a precedence rule rather than the document."
            )
        prompt_file, merged["prompt"] = _resolve_prompt_file(prompt_file, base_dir, index)
    if not isinstance(merged.get("prompt"), str):
        raise ValueError(
            f"requests[{index}]: 'prompt' (or 'prompt_file') is required and must be a string."
        )

    action_file = None
    if isinstance(merged.get("extra_params"), dict):
        extra = dict(merged["extra_params"])
        merged["extra_params"] = extra
        action_file = extra.pop("action_file", None)
        if action_file is not None:
            if extra.get("action") is not None:
                raise ValueError(
                    f"requests[{index}]: set extra_params 'action' or 'action_file', not both — "
                    "which one the run measured would depend on a precedence rule."
                )
            action_file, extra["action"] = _resolve_action_file(action_file, base_dir, index)

    try:
        request = model(**merged)
    except ValidationError as e:
        raise ValueError(f"requests[{index}]: invalid request:\n{e}") from e

    located: dict[str, str] = {}
    for slot in REFERENCE_KEYS:
        reference = getattr(request, slot, None)
        if reference is not None:
            located[slot], resolved = _resolve_reference(slot, reference, base_dir, index)
            setattr(request, slot, resolved)

    if (request.width is None) != (request.height is None):
        raise ValueError(
            f"requests[{index}]: resolved width={request.width!r} height={request.height!r}; the "
            "server rejects exactly one of them (HTTP 422). Set both, or neither."
        )
    if prompt_file is not None:
        request._original["prompt_file"] = prompt_file
    if action_file is not None:
        request._original["action_file"] = action_file
    request._original.update(located)
    return request


def _document_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the document the CLI spells out.

    Each flag maps to a named part of the document -- fields to
    ``common_params``, ``--requests`` to the list -- so the CLI is a second way
    to write it, not a second way to run. Both spellings converge here, and the
    merge and every validation below have one implementation.
    """
    common: dict[str, Any] = {}
    for key in SCALAR_PARAM_FIELDS:
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


def load_workload(args: argparse.Namespace) -> VisualGenBenchWorkload:
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
        raw, base_dir = _sniff_workload_source(args.workload)
    else:
        raw, base_dir = _document_from_args(args), Path.cwd()
    if isinstance(raw, list):
        raw = {"requests": raw}
    if not isinstance(raw, dict):
        raise ValueError("A workload is a mapping, or the bare list of requests.")

    backend = _resolve_scalar("backend", args.backend, raw.get("backend"))
    if backend is None:
        raise ValueError(
            "backend is required: set 'backend' in --workload, or pass --backend. There is "
            "no default, because it selects the endpoint and so what the run measures: a "
            "checkpoint serving both modes answers the wrong one without complaining."
        )
    if backend not in WORKLOAD_MODEL:
        raise ValueError(f"backend {backend!r} is not one of {', '.join(WORKLOAD_MODEL)}.")
    try:
        document = WORKLOAD_MODEL[backend](**{**raw, "backend": backend})
    except ValidationError as e:
        raise ValueError(f"invalid workload:\n{e}") from e

    common = document.common_params.model_dump(exclude_unset=True)
    requests = [
        _resolve_request(_merge_request(common, request, index), index, base_dir, type(request))
        for index, request in enumerate(document.requests)
    ]

    if args.num_requests is not None:
        requests = _resize_requests(requests, args.num_requests)

    workload = document.model_copy(
        update={"common_params": type(document.common_params)(), "requests": requests}
    )
    _validate_workload(workload)
    return workload


# --------------------------------------------------------------------------- #
# Pre-run validations
# --------------------------------------------------------------------------- #


def _validate_edit_reference(workload: VisualGenBenchWorkload) -> None:
    """Reject an image_reference /v1/images/edits cannot take.

    Its ``image`` follows OpenAI's schema: one string with nowhere to declare a
    wire form, so the server reads it as base64 and refuses a path or a URL.
    The video route's slot takes those, and a list of them.
    """
    if workload.backend != "openai-image-edits":
        return
    for index, request in enumerate(workload.requests):
        reference = request.image_reference
        if isinstance(reference, list) or reference.format != "base64":
            raise ValueError(
                f"requests[{index}]: 'openai-image-edits' takes a single base64 image, so "
                "image_reference must be one {content, format: base64} object; "
                "/v1/images/edits accepts neither a path nor the video backend's list form."
            )


def _validate_output_type(workload: VisualGenBenchWorkload) -> None:
    """Reject an ``extra_params.output_type`` that contradicts the backend.

    Gate on the value, not the model: Cosmos3 uses image/video to select its
    mode table, while LTX-2 reuses the key for pt/pil, which carries no
    modality and must pass through untouched.
    """
    expected = MODALITY_BY_BACKEND[workload.backend]
    for index, request in enumerate(workload.requests):
        value = (request.extra_params or {}).get("output_type")
        if value in ("image", "video") and value != expected:
            raise ValueError(
                f"requests[{index}]: extra_params.output_type={value!r} contradicts backend "
                f"{workload.backend!r} (expects {expected!r})."
            )


def _validate_workload(workload: VisualGenBenchWorkload) -> None:
    _validate_edit_reference(workload)
    _validate_output_type(workload)


# --------------------------------------------------------------------------- #
# Payload construction
# --------------------------------------------------------------------------- #


def _params_dump(request: VisualGenBenchRequest) -> dict[str, Any]:
    """The generation parameters as sent, without the input layer.

    ``exclude_unset`` keeps a request's explicit ``null`` (the server treats an
    omitted field and an explicit null identically) while never inventing a
    default.
    """
    dump = request.model_dump(exclude_unset=True)
    for key in ("prompt", *REFERENCE_KEYS):
        dump.pop(key, None)
    return dump


def build_payload(
    request: VisualGenBenchRequest,
    backend: str,
    model: str,
    response_format: str,
    output_format: Optional[str],
) -> dict[str, Any]:
    """Build the HTTP body for one request on one backend.

    The frame budget goes out as ``num_frames``; the wire's ``seconds``
    alternative is derived server-side as ``int(seconds * frame_rate)``, which
    drops a frame at 25/30/50/60/120 fps.
    """
    payload: dict[str, Any] = {
        "prompt": request.prompt,
        "model": model,
        "response_format": response_format,
    }
    params = _params_dump(request)

    if backend == VIDEO_BACKEND:
        # Typed fields, so the modality is named rather than sniffed.
        for slot in REFERENCE_KEYS:
            reference = getattr(request, slot)
            if reference is None:
                continue
            payload[slot] = (
                [item.model_dump(exclude_none=True) for item in reference]
                if isinstance(reference, list)
                else reference.model_dump(exclude_none=True)
            )
        if output_format is not None:
            payload["format"] = output_format
    elif backend == "openai-image-edits":
        payload["image"] = request.image_reference.content
        if output_format is not None:
            # ImageEditRequest's canonical name; ``format`` is only an alias.
            payload["output_format"] = output_format
    elif output_format is not None:
        payload["format"] = output_format

    wire = WIRE_MODEL[backend].model_fields
    payload.update({name if name in wire else WIRE_ALIAS[name]: v for name, v in params.items()})
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
    timings: dict[str, float], name: str, *, require_positive: bool, required: bool = True
) -> Optional[float]:
    """Return a Server-Timing metric, in seconds.

    An optional metric that is absent returns ``None``; one that is present is
    validated either way, so a bad value is never mistaken for an absent one.
    """
    if name not in timings:
        if not required:
            return None
        raise ValueError(f"Missing VisualGen Server-Timing metric: {name}")
    timing = timings[name]
    if not math.isfinite(timing) or timing < 0 or (require_positive and timing <= 0):
        raise ValueError(f"Invalid VisualGen Server-Timing metric {name}: {timing}")
    return timing


def _record_server_timings(headers: Any, record: VisualGenRequestRecord) -> None:
    """Fill the server-side timings from the Server-Timing header.

    ``generation`` and ``denoise`` are required. The rest are sent by the
    pipelines that measure them, so their absence leaves the series unreported
    rather than failing the request: ``total`` is omitted by the video route
    when the job carries no arrival stamp, and the two denoise-adjacent phases
    by any pipeline that does not time them.
    """
    timings = _parse_server_timing_header(headers)
    record.server_gen = _get_server_timing_metric(
        timings, VISUAL_GEN_GENERATION_TIMING, require_positive=True
    )
    record.server_denoise = _get_server_timing_metric(
        timings, VISUAL_GEN_DENOISE_TIMING, require_positive=False
    )
    for field_name, metric in (
        ("server_e2e", VISUAL_GEN_TOTAL_TIMING),
        ("server_pre_denoise", VISUAL_GEN_PRE_DENOISE_TIMING),
        ("server_post_denoise", VISUAL_GEN_POST_DENOISE_TIMING),
    ):
        setattr(
            record,
            field_name,
            _get_server_timing_metric(timings, metric, require_positive=True, required=False),
        )


async def _dispatch_image(
    session: aiohttp.ClientSession,
    api_url: str,
    payload: dict[str, Any],
    record: VisualGenRequestRecord,
) -> list[tuple[str, Any]]:
    """POST /v1/images/{generations,edits}: one leg, result in the response body.

    Returns what the response carried, for --output-media-dir to write after the
    request is timed.
    """
    async with session.post(url=api_url, json=payload, headers=_get_headers()) as response:
        if response.status != 200:
            record.error = f"HTTP {response.status}: {await response.text()}"
            return []
        body = await response.json()
        _record_server_timings(response.headers, record)

    record.success = True
    if payload["response_format"] == "path":
        record.output_paths = [item["path"] for item in body["data"]]
        return [("path", path) for path in record.output_paths]
    if payload["response_format"] == "b64_json":
        return [("bytes", base64.b64decode(item["b64_json"])) for item in body["data"]]
    return [("url", item["url"]) for item in body["data"]]


async def _dispatch_video(
    session: aiohttp.ClientSession,
    api_url: str,
    payload: dict[str, Any],
    record: VisualGenRequestRecord,
    t0: float,
    poll_interval: float,
    request_timeout: float,
) -> list[tuple[str, Any]]:
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
            return []
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
                return []
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
            return []
        if time.perf_counter() > deadline:
            record.error = (
                f"Video job {video_id} did not reach a terminal status within "
                f"{request_timeout}s (last status: {status})"
            )
            return []
        await asyncio.sleep(poll_interval)
    record.client_gen = generated_at

    async with session.get(url=f"{status_url}/content", headers=_get_headers()) as response:
        if response.status != 200:
            record.error = f"HTTP {response.status}: {await response.text()}"
            return []
        if payload["response_format"] == "path":
            record.output_paths = [(await response.json())["output_path"]]
            carried: list[tuple[str, Any]] = [("path", record.output_paths[0])]
        else:
            carried = [("bytes", await response.read())]
        _record_server_timings(response.headers, record)
    record.success = True
    return carried


async def _write_media(
    session: aiohttp.ClientSession,
    carried: list[tuple[str, Any]],
    out_dir: Path,
    index: int,
    output_format: Optional[str],
    modality: str,
) -> list[str]:
    """Put one request's media under ``out_dir`` and return where it landed.

    A ``path`` response names a file on the server, so it is copied and its
    readability is checked here: a client that cannot open it has no artifact,
    and the run should say so rather than record a locator that resolves
    nowhere. Bytes are written; a URL is fetched.

    Every filesystem call goes through a thread: a synchronous write holds the
    event loop, and the requests in flight beside it would carry that wait in
    their own latency.
    """
    await asyncio.to_thread(out_dir.mkdir, parents=True, exist_ok=True)
    written = []
    for i, (kind, value) in enumerate(carried):
        suffix = (
            f".{output_format}" if output_format else (".mp4" if modality == "video" else ".png")
        )
        if kind == "path":
            source = Path(value)
            if not source.is_file():
                raise ValueError(
                    f"--output-media-dir cannot take requests[{index}]: the server answered with "
                    f"{str(source)!r}, which this client cannot read. A path response names a file "
                    "on the server; ask for the bytes instead with --response-format "
                    f"{'file' if modality == 'video' else 'b64_json'}."
                )
            suffix = source.suffix or suffix
        destination = out_dir / f"{index:04d}_{i}{suffix}"
        if kind == "path":
            await asyncio.to_thread(shutil.copyfile, source, destination)
        elif kind == "bytes":
            await asyncio.to_thread(destination.write_bytes, value)
        else:
            async with session.get(url=value, headers=_get_headers()) as response:
                if response.status != 200:
                    raise ValueError(
                        f"--output-media-dir cannot take requests[{index}]: GET {value} returned "
                        f"HTTP {response.status}."
                    )
                body = await response.read()
            await asyncio.to_thread(destination.write_bytes, body)
        written.append(str(destination))
    return written


async def dispatch_request(
    session: aiohttp.ClientSession,
    backend: str,
    api_url: str,
    payload: dict[str, Any],
    record: VisualGenRequestRecord,
    benchmark_start: float,
    poll_interval: Optional[float],
    request_timeout: float,
) -> tuple[VisualGenRequestRecord, list[tuple[str, Any]]]:
    """Run one request end to end; client_e2e is recorded even on failure.

    The media the response carried goes back with the record, so the caller can
    write it after releasing its concurrency slot.
    """
    t0 = time.perf_counter()
    record.start = t0 - benchmark_start
    carried: list[tuple[str, Any]] = []
    try:
        if backend == VIDEO_BACKEND:
            carried = await _dispatch_video(
                session, api_url, payload, record, t0, poll_interval, request_timeout
            )
        else:
            carried = await _dispatch_image(session, api_url, payload, record)
    except Exception:
        record.success = False
        record.error = "".join(traceback.format_exception(*sys.exc_info()))
    finally:
        record.end = time.perf_counter() - benchmark_start
        record.client_e2e = record.end - record.start
    return record, carried


# Carried in the payload but not generation parameters; references are recorded
# separately, by locator rather than by their base64 payload.
_NON_PARAM_PAYLOAD_KEYS = frozenset({"prompt", "model", "image", *REFERENCE_KEYS})


def _make_record(
    index: int, request: VisualGenBenchRequest, payload: dict[str, Any]
) -> VisualGenRequestRecord:
    """Record the parameters as sent -- a backend drops the ones it cannot carry.

    References are recorded by locator: a video reference is tens of MB, and a
    result that carried the bytes would dwarf the numbers it annotates.
    """
    return VisualGenRequestRecord(
        index=index,
        prompt=request.prompt,
        prompt_file=request._original.get("prompt_file"),
        action_file=request._original.get("action_file"),
        params={k: v for k, v in payload.items() if k not in _NON_PARAM_PAYLOAD_KEYS},
        **{slot: request._original.get(slot) for slot in REFERENCE_KEYS},
    )


async def _paced(count: int, request_rate: float, burstiness: float) -> AsyncGenerator[int, None]:
    """Yield request indices at ``request_rate``, spaced by a gamma interval.

    Mirrors ``benchmark_serving.py``'s ``get_request``: the mean interval is
    ``1 / request_rate`` whatever the burstiness, which only sets the spread
    around it.
    """
    if burstiness <= 0:
        raise ValueError(f"--burstiness must be positive, got {burstiness}.")
    theta = 1.0 / (request_rate * burstiness) if request_rate != float("inf") else 0.0
    for index in range(count):
        yield index
        if request_rate == float("inf"):
            continue
        await asyncio.sleep(np.random.gamma(shape=burstiness, scale=theta))


async def benchmark(
    *,
    backend: str,
    base_url: str,
    model: str,
    workload: VisualGenBenchWorkload,
    response_format: str,
    output_format: Optional[str],
    disable_tqdm: bool,
    max_concurrency: Optional[int],
    no_test_input: bool,
    request_timeout: float,
    poll_interval: Optional[float],
    request_rate: float,
    burstiness: float,
    output_media_dir: Optional[Path],
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
            record, carried = await dispatch_request(*args, **kwargs)
        else:
            async with semaphore:
                record, carried = await dispatch_request(*args, **kwargs)
        # Outside the slot: keeping the media is the client's own bookkeeping,
        # and holding the slot for it would cap concurrency below what was asked.
        if output_media_dir is not None and record.success and carried:
            record.output_paths = await _write_media(
                args[0],
                carried,
                output_media_dir,
                record.index,
                output_format,
                MODALITY_BY_BACKEND[backend],
            )
        if pbar:
            pbar.update(1)
        return record

    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(
        trust_env=True,
        timeout=timeout,
        connector=aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True),
    ) as session:
        if not no_test_input:
            print("Starting initial single prompt test run...")
            test_record, _ = await dispatch_request(
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
        print(f"Traffic request rate: {request_rate}")
        print(f"Burstiness factor: {burstiness}")

        benchmark_start = time.perf_counter()
        tasks = []
        async for index in _paced(len(payloads), request_rate, burstiness):
            tasks.append(
                asyncio.create_task(
                    limited_dispatch(
                        session,
                        backend,
                        api_url,
                        payloads[index],
                        _make_record(index, workload.requests[index], payloads[index]),
                        benchmark_start,
                        poll_interval,
                        request_timeout,
                    )
                )
            )
        records: list[VisualGenRequestRecord] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    # The window closes when the last request did, which each record stamped
    # before anything was written to disk. Reading the clock here instead would
    # put --output-media-dir's writes into the duration that request_throughput
    # and the output rates divide by.
    return records, max(record.end for record in records)


# --------------------------------------------------------------------------- #
# Aggregation and output
# --------------------------------------------------------------------------- #


def _percentile_key(percentile: float) -> str:
    return f"p{int(percentile) if int(percentile) == percentile else percentile}"


def _samples(records: list[VisualGenRequestRecord], name: str) -> list[float]:
    return [getattr(r, name) for r in records if r.success and getattr(r, name) is not None]


def _stats(samples: list[float], selected_percentiles: list[float]) -> Optional[dict[str, Any]]:
    """Aggregate one timing series, or ``None`` when nothing reported it.

    An optional metric the server omits has no statistics. Zero-filling the
    block would publish that absence as a measured zero, which a consumer
    cannot tell apart from a real one.
    """
    if not samples:
        return None
    return {
        **{column: float(func(samples)) for column, func in STAT_FUNCS.items()},
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
        count = sum(int(record.params.get("n", 1)) for record in done)
    if count is None:
        return key, None
    return key, count / duration if duration > 0 else 0.0


def _record_json(record: VisualGenRequestRecord) -> dict[str, Any]:
    data: dict[str, Any] = {
        "index": record.index,
        "success": record.success,
        "prompt": record.prompt,
        "params": record.params,
        "start": record.start,
        "end": record.end,
        "client_e2e": record.client_e2e,
        "client_gen": record.client_gen,
        **{name: getattr(record, name) for name in SERVER_TIMING_FIELDS},
        "poll_count": record.poll_count,
    }
    if record.prompt_file is not None:
        data["prompt_file"] = record.prompt_file
    if record.action_file is not None:
        data["action_file"] = record.action_file
    for slot in REFERENCE_KEYS:
        value = getattr(record, slot)
        if value is not None:
            data[slot] = value
    data["output_paths"] = record.output_paths
    data["error"] = record.error
    return data


_STATS_SHAPE = (
    "{mean, median, std, min, max, percentiles} over the run's requests, one sample per "
    "request, with the percentiles --metric-percentiles asked for. Null when nothing "
    "reported the metric."
)


class VisualGenBenchResult(StrictBaseModel):
    """The result JSON, and what ``--save-result`` writes.

    Declared in the order it is emitted. A field left unset is absent from the
    file rather than null, so a reader can tell "this run had no such thing"
    from "nothing reported it": the rate key is whichever the backend produces,
    ``per_gpu_throughput`` needs ``--num-gpus``, ``gen_latency`` is video only,
    and ``timings`` and ``requests`` need ``--save-detailed``.
    """

    schema_version: int = Field(
        description="Shape of this file. A file without the key predates it: its generation "
        "config lived in flags and its latency keys carried other names."
    )
    date: str = Field(description="When the run finished, as YYYYmmdd-HHMMSS local time.")
    backend: str = Field(description="Which endpoint the run measured.")
    model: str = Field(description="The model id the requests carried.")
    duration: float = Field(description="Seconds from the first send to the last finish.")
    config: dict[str, Any] = Field(
        description="The run settings this result was produced under, so a stored file says "
        "how it was measured."
    )
    total_requests: int = Field(description="Requests sent.")
    completed: int = Field(
        description="Requests that succeeded. Below total_requests, the run is not a result."
    )
    request_throughput: float = Field(
        description="Completed requests over the run's duration, in req/s."
    )
    frames_per_second: Optional[float] = Field(
        default=None, description="Frames produced over the run's duration. Video backends."
    )
    images_per_second: Optional[float] = Field(
        default=None, description="Images produced over the run's duration. Image backends."
    )
    e2e_latency: Optional[dict[str, Any]] = Field(
        default=None,
        description="Whole-request latency as the caller sees it, network and this client's "
        f"own reads included. {_STATS_SHAPE}",
    )
    per_gpu_throughput: Optional[float] = Field(
        default=None, description="request_throughput divided by --num-gpus."
    )
    gen_latency: Optional[dict[str, Any]] = Field(
        default=None,
        description="Latency to the job first reporting 'postprocessing' or 'completed', so "
        "it stops before the media encode that e2e_latency includes. Video only, and its "
        f"granularity is --poll-interval. {_STATS_SHAPE}",
    )
    timings: Optional[dict[str, Any]] = Field(
        default=None,
        description="The server-measured metrics, one per server_* field of a request record, "
        f"read from the Server-Timing response header. {_STATS_SHAPE}",
    )
    requests: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="One record per request. A heterogeneous run cannot be attributed "
        "without them.",
    )


def build_visual_gen_result(
    *,
    backend: str,
    model: str,
    duration: float,
    records: list[VisualGenRequestRecord],
    selected_percentiles: list[float],
    config: dict[str, Any],
    save_detailed: bool,
    num_gpus: Optional[int] = None,
) -> dict[str, Any]:
    """Assemble the result JSON; stdout is printed from this same dict.

    Returned as a dict rather than the model so the conditional keys stay
    absent: ``VisualGenBenchResult`` declares and documents every key, and the
    unset ones are dropped here.
    """
    completed = sum(1 for record in records if record.success)
    rate_key, rate = _output_rate(records, backend, duration)
    throughput = completed / duration if duration > 0 else 0.0

    result = VisualGenBenchResult(
        schema_version=RESULT_SCHEMA_VERSION,
        date=datetime.now().strftime("%Y%m%d-%H%M%S"),
        backend=backend,
        model=model,
        duration=duration,
        config=config,
        total_requests=len(records),
        completed=completed,
        request_throughput=throughput,
        **{rate_key: rate},
        e2e_latency=_stats(_samples(records, "client_e2e"), selected_percentiles),
        per_gpu_throughput=throughput / num_gpus if num_gpus else None,
        gen_latency=(
            _stats(_samples(records, "client_gen"), selected_percentiles)
            if backend == VIDEO_BACKEND
            else None
        ),
        timings=(
            {
                name: _stats(_samples(records, name), selected_percentiles)
                for name in SERVER_TIMING_FIELDS
            }
            if save_detailed
            else None
        ),
        requests=[_record_json(record) for record in records] if save_detailed else None,
    )
    absent = {name for name in ("frames_per_second", "images_per_second") if name != rate_key}
    if not num_gpus:
        absent.add("per_gpu_throughput")
    if backend != VIDEO_BACKEND:
        absent.add("gen_latency")
    if not save_detailed:
        absent |= {"timings", "requests"}
    return result.model_dump(exclude=absent)


def _apply_metadata(result: dict[str, Any], metadata: Optional[list[str]]) -> None:
    """Add each ``KEY=VALUE`` to ``result``, refusing to restate one of its own.

    The run's own fields are past validation by the time this writes, and the
    partial-run check reads ``completed`` and ``total_requests`` back out of
    this dict, so an overwrite would both falsify the saved result and decide
    whether the run reports its own failure.
    """
    for item in metadata or []:
        if "=" not in item:
            raise ValueError("Invalid metadata format. Please use KEY=VALUE format.")
        key, value = item.split("=", 1)
        key = key.strip()
        if key in result:
            raise ValueError(
                f"--metadata {key}=... would overwrite the run's own {key!r}. "
                "Metadata annotates a result; it cannot restate one."
            )
        result[key] = value.strip()


def print_visual_gen_results(result: dict[str, Any], selected_percentiles: list[float]) -> None:
    is_video = result["backend"] == VIDEO_BACKEND
    failed = result["total_requests"] - result["completed"]
    rate = result["frames_per_second"] if is_video else result["images_per_second"]

    summary = [
        ("Backend:", result["backend"]),
        ("Model:", result["model"]),
        (
            "Total / Successful / Failed:",
            f"{result['total_requests']} / {result['completed']} / {failed}",
        ),
        ("Benchmark duration (s):", f"{result['duration']:.2f}"),
        ("Request throughput (req/s):", f"{result['request_throughput']:.4f}"),
        (
            "Frames per second:" if is_video else "Images per second:",
            "n/a" if rate is None else f"{rate:.2f}",
        ),
        ("Max concurrency:", str(result["config"]["max_concurrency"])),
    ]

    percentile_keys = [_percentile_key(p) for p in selected_percentiles]
    header = ["Timing (s)", *STAT_COLUMNS, *percentile_keys]
    rows = [
        [
            name,
            *(f"{block[column]:.3f}" for column in STAT_COLUMNS),
            *(f"{block['percentiles'][key]:.3f}" for key in percentile_keys),
        ]
        for name in ("e2e_latency", "gen_latency")
        if (block := result.get(name)) is not None
    ]
    # Measured from what this run prints: --metric-percentiles decides the column
    # count, and a latency has no bound to size a column against.
    widths = [max(len(cell) for cell in column) + 2 for column in zip(header, *rows)]
    width = sum(widths)

    print("{s:{c}^{n}}".format(s=" Benchmark Result (VisualGen) ", n=width, c="="))
    label_width = max(len(label) for label, _ in summary) + 2
    for label, value in summary:
        print(f"{label:<{label_width}}{value}")

    if failed:
        print("=" * width)
        print(f"  !!! {failed} FAILED REQUESTS - CHECK LOG FOR ERRORS !!!")
        print("=" * width)

    print("-" * width)
    for row in (header, *rows):
        print(
            f"{row[0]:<{widths[0]}}"
            + "".join(f"{cell:>{w}}" for cell, w in zip(row[1:], widths[1:]))
        )
    print("=" * width)


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

    conn_group = parser.add_argument_group("Connection", "Where the server is, and what it serves.")
    conn_group.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
    conn_group.add_argument("--port", type=int, default=8000, help="Server port.")
    conn_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id or checkpoint path sent as the request 'model' field and used to "
        "label results, so a stored result names what produced it. Default: the id "
        "reported by GET /v1/models; passing it cross-checks against that id rather "
        "than replacing it.",
    )

    workload_group = parser.add_argument_group(
        "Workload",
        "What the run sends. The document arrives one of two ways and never both: "
        "--workload, or the field flags below, which are its common_params, with "
        "--requests carrying its requests list. --backend and --num-requests apply to "
        "whichever way it arrived.",
    )
    workload_group.add_argument(
        "--workload",
        type=str,
        default=None,
        help="Workload document: a YAML/JSON file path, or inline content starting with "
        "'{' (full mapping) or '[' (bare requests list) -- a path starting with either "
        "character is not addressable. Top-level keys: backend, common_params, requests.",
    )
    workload_group.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=list(BACKEND_ENDPOINTS),
        help="Which endpoint to measure. Supplies the document's 'backend' key when it omits "
        "one, and disagreeing with it is an error; required from one of the two. A "
        "checkpoint serving both modes answers the wrong one without complaining.",
    )
    workload_group.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Resize the resolved requests list to exactly this many, cycling it in "
        "order or truncating it. Default: send the document as written.",
    )
    # One flag per document key, same name, so a command line and a file describe a
    # workload identically.
    workload_group.add_argument(
        "--requests",
        type=str,
        default=None,
        help="The requests list as JSON, in the document's own form, e.g. "
        '\'[{"prompt": "a fox"}, {"prompt": "a cat", "seed": 7}]\'. Each entry '
        "overrides the fields below per key. Required when spelling the document out, "
        "as it is in a file.",
    )
    for name, kind in SCALAR_PARAM_FIELDS.items():
        workload_group.add_argument(
            f"--{name.replace('_', '-')}",
            type=kind,
            default=None,
            help=VisualGenParams.model_fields[name].description,
        )
    workload_group.add_argument(
        "--extra-params", type=str, default=None, help="Per-pipeline parameters, as a JSON object."
    )

    traffic_group = parser.add_argument_group("Traffic", "When requests are issued.")
    traffic_group.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum requests in flight (default: unbounded).",
    )
    traffic_group.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Arrival rate in req/s (default: %(default)s, which sends every request at "
        "once). It paces when a request is created; --max-concurrency caps how many run.",
    )
    traffic_group.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Spread of the arrival intervals, and only in effect when --request-rate is "
        "finite (default: %(default)s, an exponential interval, so a Poisson process). A "
        "lower value (0 < burstiness < 1) results in more bursty requests, while a higher "
        "value (burstiness > 1) results in a more uniform arrival of requests.",
    )

    exec_group = parser.add_argument_group(
        "Execution", "How the client drives the run, and how the media comes back."
    )
    exec_group.add_argument(
        "--no-test-input",
        action="store_true",
        help="Skip the single probe request sent before the measured run. The probe is "
        "not counted, and it fails fast on a workload the server rejects.",
    )
    exec_group.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help=f"Job status poll interval in seconds for {VIDEO_BACKEND} "
        "(default: %(default)s). It is the granularity of gen_latency and e2e_latency; "
        "the image backends are synchronous and ignore it.",
    )
    exec_group.add_argument(
        "--request-timeout",
        type=float,
        default=6 * 60 * 60,
        help="Request timeout in seconds (default: 6 hours).",
    )
    exec_group.add_argument("--disable-tqdm", action="store_true", help="Disable progress bar.")
    exec_group.add_argument(
        "--response-format",
        type=str,
        default="path",
        help="How the server returns media (default: %(default)s): 'path' returns a "
        "locator, the others return the bytes. The backends otherwise accept 'file' "
        "(video, its own default) and 'url' / 'b64_json' (images, default 'url'). "
        "Run-level: mixing transport modes within one run makes the aggregate latency "
        "incomparable.",
    )
    exec_group.add_argument(
        "--format",
        type=str,
        default=None,
        help="Encoding the server writes: mp4/avi/auto for video, png/webp/jpeg for "
        "images. Default: the server's, which for video is 'auto' -- without ffmpeg "
        "that is AVI/MJPEG, a different encode inside the measured window.",
    )

    results_group = parser.add_argument_group("Results", "What the run writes down.")
    results_group.add_argument(
        "--save-result",
        action="store_true",
        help="Write the result JSON. Without it the run only prints.",
    )
    results_group.add_argument(
        "--save-detailed",
        action="store_true",
        help="Add the timings.server_* metrics and a per-request record to the result "
        "JSON. A heterogeneous run cannot be attributed without them.",
    )
    results_group.add_argument(
        "--result-dir", type=str, default=None, help="Directory for result files."
    )
    results_group.add_argument(
        "--result-filename", type=str, default=None, help="Custom result filename."
    )
    results_group.add_argument(
        "--output-media-dir",
        type=str,
        default=None,
        help="Write each successful request's media here as {index}_{i}{ext}, and record "
        "where it landed in output_paths. A 'path' response is copied from the server's "
        "file, which the client has to be able to read; the other transports are written "
        "from the bytes they carry, a URL after fetching it. Writing happens outside the "
        "measured window.",
    )
    results_group.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="GPUs the server runs on, recorded in the result and used to divide "
        "request_throughput into per_gpu_throughput. The same model and workload at 1, 4 "
        "or 8 GPUs are different measurements, and the server reports no topology of its "
        "own. Its value is the product of the server's parallel sizes.",
    )
    results_group.add_argument(
        "--metric-percentiles",
        type=str,
        default="50,90,99",
        help="Comma-separated percentile values (default: '50,90,99').",
    )
    results_group.add_argument(
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
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            output_media_dir=Path(args.output_media_dir) if args.output_media_dir else None,
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

    if args.num_gpus is not None and args.num_gpus < 1:
        raise ValueError(f"--num-gpus must be >= 1, got {args.num_gpus}.")

    config: dict[str, Any] = {
        "num_requests": len(records),
        "num_gpus": args.num_gpus,
        "max_concurrency": args.max_concurrency,
        "output_media_dir": args.output_media_dir,
        "request_rate": args.request_rate if args.request_rate != float("inf") else "inf",
        "burstiness": args.burstiness,
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
        num_gpus=args.num_gpus,
    )
    print_visual_gen_results(result, selected_percentiles)

    if args.save_result:
        _apply_metadata(result, args.metadata)

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

    # After the result is written: the numbers of a partial run are still worth
    # keeping, but a caller must be able to see it was partial without parsing them.
    if result["completed"] != result["total_requests"]:
        raise SystemExit(
            f"{result['total_requests'] - result['completed']} of "
            f"{result['total_requests']} requests failed; this is not a result."
        )


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
