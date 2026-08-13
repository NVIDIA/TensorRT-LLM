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
"""LLM API configuration capture for usage telemetry."""

from __future__ import annotations

import hashlib
import json
import math
import types
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from tensorrt_llm.usage.config import TelemetryField

CAPTURE_VERSION = "2"
FIELD_POLICY_VERSION = "2"
API_CONTRACT_VERSION = "0.2.0"
CAPTURE_SOURCE = "effective_validated_llm_args"

# Cap total serialized bytes of llmApiConfigJson. The wire field is unbounded and
# the reporter is fail-silent, so an oversized payload is dropped whole by the
# endpoint; truncate and flag instead. Conservative bound until the endpoint limit
# is confirmed.
MAX_CONFIG_BYTES = 16384

_TELEMETRY_EXTRA_KEY = "telemetry"
_TRTLLM_JSON_SCHEMA_EXTRA_ATTR = "_trtllm_json_schema_extra"
_APPROVED_CONVERTERS = {"allowlist"}

# Per-sequence cap, applied recursively so each inner list of a nested
# List[List[int]] is bounded independently. 256 sits above the longest realistic
# captured sequence (~200 per-layer entries) yet still bounds a runaway list.
MAX_SEQ_ITEMS = 256


class _CaptureState:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.excluded_field_count = 0
        self.unsafe_excluded = False
        self.sequence_truncated = False
        self.payload_truncated = False


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _digest(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _is_pydantic_model(value: Any) -> bool:
    return isinstance(value, BaseModel)


def _none_type() -> type[None]:
    return type(None)


def _unwrap_annotated(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _is_union(annotation: Any) -> bool:
    return get_origin(annotation) in {Union, types.UnionType}


def _unwrap_optional(annotation: Any) -> Any:
    annotation = _unwrap_annotated(annotation)
    if not _is_union(annotation):
        return annotation
    branches = [arg for arg in get_args(annotation) if arg is not _none_type()]
    if len(branches) == 1:
        return branches[0]
    return annotation


def _is_literal(annotation: Any) -> bool:
    return get_origin(annotation) is Literal


def _is_enum_annotation(annotation: Any) -> bool:
    try:
        return isinstance(annotation, type) and issubclass(annotation, Enum)
    except TypeError:
        return False


def _is_path_annotation(annotation: Any) -> bool:
    try:
        return isinstance(annotation, type) and issubclass(annotation, Path)
    except TypeError:
        return False


def _is_callable_annotation(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return annotation is Callable or origin is Callable


def _is_safe_annotation_branch(annotation: Any) -> bool:
    annotation = _unwrap_annotated(annotation)
    origin = get_origin(annotation)

    if annotation is Any:
        return False
    if annotation is _none_type():
        return True
    if _is_literal(annotation):
        return True
    if _is_enum_annotation(annotation):
        return True
    if annotation in {bool, int, float}:
        return True
    if annotation is str or annotation is object:
        return False
    if _is_path_annotation(annotation) or _is_callable_annotation(annotation):
        return False
    if _is_union(annotation):
        return all(_is_safe_annotation_branch(arg) for arg in get_args(annotation))
    if origin in {list, tuple, set}:
        args = get_args(annotation)
        return bool(args) and all(_is_safe_annotation_branch(arg) for arg in args)
    if origin is dict:
        return False
    return False


def _union_needs_converter(annotation: Any) -> bool:
    annotation = _unwrap_annotated(annotation)
    if not _is_union(annotation):
        return False
    branches = [arg for arg in get_args(annotation) if arg is not _none_type()]
    if len(branches) <= 1:
        return False
    return not all(_is_safe_annotation_branch(arg) for arg in branches)


def _normalize_metadata(metadata: Any) -> dict[str, Any] | None:
    if metadata is None:
        return None
    if metadata is False:
        return {"exclude": True}
    if metadata is True:
        return {"kind": "value"}
    if isinstance(metadata, TelemetryField):
        return metadata.as_json_schema_extra()
    if isinstance(metadata, dict):
        return dict(metadata)
    return None


def _get_telemetry_metadata(field_info: Any) -> dict[str, Any] | None:
    json_schema_extra = getattr(field_info, "json_schema_extra", None)
    if callable(json_schema_extra):
        json_schema_extra = getattr(json_schema_extra, _TRTLLM_JSON_SCHEMA_EXTRA_ATTR, None)
    if not isinstance(json_schema_extra, dict):
        return None
    return _normalize_metadata(json_schema_extra.get(_TELEMETRY_EXTRA_KEY))


def _converter_is_approved(metadata: dict[str, Any]) -> bool:
    return metadata.get("converter") in _APPROVED_CONVERTERS


def _is_explicit_exclude(metadata: dict[str, Any] | None) -> bool:
    return bool(metadata) and metadata.get("exclude") is True


def _metadata_has_allowlist(metadata: dict[str, Any]) -> bool:
    return _converter_is_approved(metadata) or metadata.get("allowed_values") is not None


def derive_kind(annotation: Any, metadata: dict[str, Any]) -> str:
    """Derive a telemetry field's kind from its annotation and metadata.

    Categorical iff the Optional-unwrapped annotation is a Literal or Enum, or it
    carries an allowlist; otherwise a plain value. Any registered kind is ignored
    so kind stays annotation-driven.
    """
    if _metadata_has_allowlist(metadata):
        return "categorical"
    unwrapped = _unwrap_optional(annotation)
    if _is_literal(unwrapped) or _is_enum_annotation(unwrapped):
        return "categorical"
    return "value"


def _annotation_is_capture_safe(annotation: Any) -> bool:
    return _is_safe_annotation_branch(annotation)


@dataclass(frozen=True)
class _ManifestEntry:
    path: str
    annotation: Any  # real type object — needed by the sanitizer
    kind: str
    converter: str
    allowed_values: tuple[str, ...]
    metadata: dict[str, Any]  # normalized metadata (allowlist) for the sanitizer


def _domain_values(annotation: Any, metadata: dict[str, Any]) -> list[str]:
    """Compute a field's allowed-value domain.

    Explicit allowlist wins; else Literal args AND Enum members found anywhere in
    the annotation tree, order-preserving + deduped. Mirrors what _sanitize_value
    would emit for an Enum (str value, else name).
    """
    allowlist = metadata.get("allowed_values") if metadata else None
    if isinstance(allowlist, (list, tuple, set)):
        return [str(v) for v in allowlist]

    values: list[str] = []

    def rec(ann: Any) -> None:
        ann = _unwrap_annotated(ann)
        if _is_literal(ann):
            for v in get_args(ann):
                if isinstance(v, (bool, int, float, str)) or v is None:
                    values.append(str(v))
            return
        if _is_enum_annotation(ann):
            for member in ann:
                values.append(member.value if isinstance(member.value, str) else member.name)
            return
        if _is_union(ann) or get_origin(ann) in {list, tuple, set}:
            for arg in get_args(ann):
                rec(arg)

    rec(annotation)
    seen: list[str] = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


def _nested_models(annotation: Any) -> list[type]:
    """Every BaseModel reachable in an annotation tree.

    Covers Optional / Union / discriminated-union arms / list|tuple|set element
    types. dict is NOT traversed (keys/values are not captured).
    """
    out: list[type] = []

    def rec(ann: Any) -> None:
        ann = _unwrap_annotated(ann)
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            out.append(ann)
            return
        if _is_union(ann) or get_origin(ann) in {list, tuple, set}:
            for arg in get_args(ann):
                rec(arg)

    rec(annotation)
    deduped: list[type] = []
    for m in out:
        if m not in deduped:
            deduped.append(m)
    return deduped


def _defining_class(cls: type, field_name: str) -> str:
    for klass in cls.__mro__:
        if field_name in getattr(klass, "__annotations__", {}):
            return f"{klass.__name__}.{field_name}"
    return f"{cls.__name__}.{field_name}"


def _field_is_selected(annotation: Any, metadata: dict[str, Any] | None) -> bool:
    if _is_explicit_exclude(metadata):
        return False
    if _annotation_is_capture_safe(annotation):
        return True
    if metadata is not None and _converter_is_approved(metadata):
        return True
    return False


def build_capture_manifest(model_cls: type[BaseModel]) -> list[_ManifestEntry]:
    """Walk real type objects and emit the complete capturable manifest.

    The single source of truth. Type-safe annotations auto-enroll; str/Any
    allowlist escape hatches opt in; telemetry=False opts out. Recurses into
    statically reachable nested BaseModels with a cycle guard. Collapses
    duplicate keys (shared union-arm base fields): keeps the first by
    (key, defining_class), unions allowed_values across arms, and FAILS if two
    arms give a key a different kind.
    """
    rows: list[dict[str, Any]] = []

    def walk(cls: type, prefix: str, stack: tuple) -> None:
        if cls in stack:
            return
        for fname, finfo in cls.model_fields.items():
            key = f"{prefix}.{fname}" if prefix else fname
            ann = finfo.annotation
            meta = _get_telemetry_metadata(finfo)
            if _field_is_selected(ann, meta):
                normalized = meta if (meta and not _is_explicit_exclude(meta)) else {}
                rows.append(
                    {
                        "key": key,
                        "defining": _defining_class(cls, fname),
                        "annotation": ann,
                        "kind": derive_kind(ann, normalized),
                        "converter": str(normalized.get("converter", "")),
                        "allowed": _domain_values(ann, normalized),
                        "metadata": normalized,
                    }
                )
            if not _is_explicit_exclude(meta):
                for sub in _nested_models(ann):
                    walk(sub, key, (*stack, cls))

    walk(model_cls, "", ())

    rows.sort(key=lambda r: (r["key"], r["defining"]))
    first: dict[str, dict] = {}
    union_allowed: dict[str, list[str]] = {}
    for r in rows:
        if r["key"] not in first:
            first[r["key"]] = r
        elif first[r["key"]]["kind"] != r["kind"]:
            raise ValueError(
                f"telemetry manifest: key '{r['key']}' has conflicting kinds "
                f"across union arms: {first[r['key']]['kind']} vs {r['kind']}"
            )
        seen = union_allowed.setdefault(r["key"], [])
        for v in r["allowed"]:
            if v not in seen:
                seen.append(v)

    entries = [
        _ManifestEntry(
            path=key,
            annotation=r["annotation"],
            kind=r["kind"],
            converter=r["converter"],
            allowed_values=tuple(union_allowed[key]),
            metadata=r["metadata"],
        )
        for key, r in first.items()
    ]
    entries.sort(key=lambda e: e.path)
    return entries


def manifest_rows(model_cls: type[BaseModel]) -> list[dict[str, Any]]:
    """Serializable, human-legible projection of build_capture_manifest.

    Used by the committed golden, the docs renderer, and the
    capture_manifest_digest.
    """
    return [
        {
            "path": e.path,
            "annotation": _annotation_repr(e.annotation),
            "kind": e.kind,
            "converter": e.converter,
            "allowed_values": list(e.allowed_values),
        }
        for e in build_capture_manifest(model_cls)
    ]


def golden_manifest() -> dict[str, list[dict[str, Any]]]:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    return {
        "TorchLlmArgs": manifest_rows(TorchLlmArgs),
    }


def _sanitize_allowlist(value: Any, metadata: dict[str, Any]) -> tuple[bool, Any]:
    """Capture value only when it matches field-owned allowed values."""
    allowed_values = metadata.get("allowed_values")
    if not isinstance(allowed_values, (list, tuple, set)):
        return False, None
    candidates = [value]
    if isinstance(value, Enum):
        candidates.append(value.value)
        candidates.append(value.name)
        candidates.append(value.name.lower())

    for candidate in candidates:
        if candidate in allowed_values and (
            isinstance(candidate, (bool, int, float, str)) or candidate is None
        ):
            return True, candidate
    return False, None


def _sanitize_literal(value: Any, annotation: Any) -> tuple[bool, Any]:
    """Capture only values declared by Literal annotation."""
    allowed = get_args(annotation)
    if value in allowed:
        return True, value
    return False, None


def _sanitize_sequence(
    value: Any,
    annotation: Any,
    metadata: dict[str, Any],
    state: _CaptureState | None = None,
) -> tuple[bool, Any]:
    """Sanitize homogeneous sequence values. Reject whole sequence on one bad item.

    Captured items are capped at MAX_SEQ_ITEMS. The cap rides the recursive
    _sanitize_value call, so each inner list of a nested sequence is bounded
    independently. When any sequence is clipped, state.sequence_truncated is
    set so the metadata reports the truncation honestly.
    """
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin not in {list, tuple, set}:
        element_annotation = Any
    else:
        args = get_args(annotation)
        # Homogeneous only. tuple[int, str] uses first annotation and rejects on
        # mismatch. Fail closed if future telemetry field adds heterogeneous tuple.
        element_annotation = args[0] if args else Any

    sanitized = []
    for item in value:
        item_safe, item_value = _sanitize_value(item, element_annotation, metadata, state)
        if not item_safe:
            return False, None
        sanitized.append(item_value)
    if origin is set:
        sanitized = sorted(sanitized, key=_canonical_json)
    if len(sanitized) > MAX_SEQ_ITEMS:
        sanitized = sanitized[:MAX_SEQ_ITEMS]
        if state is not None:
            state.sequence_truncated = True
    return True, sanitized


def _sanitize_value(
    value: Any,
    annotation: Any,
    metadata: dict[str, Any],
    state: _CaptureState | None = None,
) -> tuple[bool, Any]:
    """Return telemetry-safe primitive value, else exclude it.

    Bare strings are unsafe unless Literal or allowlist-converted. Exclusion is
    visible through unsafe_excluded metadata.
    """
    has_converter = _converter_is_approved(metadata)
    if _union_needs_converter(annotation) and not has_converter:
        return False, None
    if not has_converter and not _annotation_is_capture_safe(annotation):
        return False, None
    annotation = _unwrap_optional(annotation)

    # None: Optional field unset -> capture as null, regardless of converter.
    # Must precede the allowlist branch, else None on an Optional allowlist field
    # fails the allowlist and falsely flips unsafe_excluded.
    if value is None:
        return True, None

    if has_converter:
        return _sanitize_allowlist(value, metadata)
    if isinstance(value, Enum):
        enum_value = value.value
        if isinstance(enum_value, str):
            return True, enum_value
        # Enum.name always str. Prefer stable names for int-valued enums.
        return True, value.name
    # bool before int. Python bool is int subclass; keep True/False not 1/0.
    if isinstance(value, bool):
        return True, value
    if isinstance(value, int) and not isinstance(value, bool):
        return True, value
    if isinstance(value, float):
        # Reject nan/inf. json.dumps emits the bare NaN/Infinity tokens for
        # non-finite floats, which are invalid JSON and break downstream
        # parsing and digest stability.
        if not math.isfinite(value):
            return False, None
        return True, value
    if isinstance(value, str):
        if _is_literal(annotation):
            return _sanitize_literal(value, annotation)
        # Bare str can be path/secret/user text. Require Literal or allowlist.
        return False, None
    if isinstance(value, Path):
        return False, None
    if isinstance(value, (list, tuple, set)):
        return _sanitize_sequence(value, annotation, metadata, state)
    return False, None


def _annotation_repr(annotation: Any) -> str:
    text = repr(annotation)
    return text.replace("typing.", "")


def _schema_digest(model_cls: type[BaseModel]) -> str:
    schema_fields = []
    for field_name, field_info in sorted(model_cls.model_fields.items()):
        schema_fields.append(
            {
                "path": field_name,
                "annotation": _annotation_repr(field_info.annotation),
                "required": field_info.is_required(),
            }
        )
    return _digest({"class": model_cls.__name__, "fields": schema_fields})


def _resolve_path(instance: BaseModel, path: str) -> tuple[bool, Any]:
    """Resolve a dotted manifest path against a live instance.

    Returns (present, value). Skips when a parent segment is missing/None or is
    not a pydantic model (unset config, or a discriminated-union arm that isn't
    the active one). A present leaf whose value is None resolves as (True, None).
    """
    segments = path.split(".")
    obj: Any = instance
    for seg in segments[:-1]:
        if not _is_pydantic_model(obj):
            return False, None
        if seg not in obj.__class__.model_fields:
            return False, None
        obj = getattr(obj, seg, None)
        if obj is None:
            return False, None
    leaf = segments[-1]
    if not _is_pydantic_model(obj) or leaf not in obj.__class__.model_fields:
        return False, None
    return True, getattr(obj, leaf, None)


def _truncate_to_budget(values: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Keep deterministically (sorted keys) as many fields as fit MAX_CONFIG_BYTES."""
    kept: dict[str, Any] = {}
    for key in sorted(values):
        trial = dict(kept)
        trial[key] = values[key]
        if len(_canonical_json(trial).encode("utf-8")) > MAX_CONFIG_BYTES:
            break
        kept = trial
    return kept, _canonical_json(kept)


def _failure_meta(args_class: str = "") -> dict[str, Any]:
    """Metadata for capture failure. One shape used by collector and reporter."""
    return {
        "api_contract_version": API_CONTRACT_VERSION,
        "args_class": args_class,
        "capture_manifest_digest": "",
        "capture_succeeded": False,
        "capture_version": CAPTURE_VERSION,
        "capturable_field_count": 0,
        "captured_field_count": 0,
        "excluded_field_count": 0,
        "field_policy_version": FIELD_POLICY_VERSION,
        "payload_truncated": False,
        "schema_digest": "",
        "sequence_truncated": False,
        "source": CAPTURE_SOURCE,
        "unsafe_excluded": False,
    }


def _failure_llm_api_config_payloads(args_class: str = "") -> tuple[str, str]:
    """Return empty config plus canonical failure metadata JSON."""
    return "{}", _canonical_json(_failure_meta(args_class=args_class))


def collect_llm_api_config_payloads(llm_args: Any) -> tuple[str, str]:
    """Return sanitized LLM API config and capture metadata JSON strings.

    Manifest-driven: capture exactly the keys build_capture_manifest lists for
    this class, so the runtime can never emit a key absent from the committed
    golden (runtime_keys subset of manifest_keys, by construction).
    """
    try:
        if not _is_pydantic_model(llm_args):
            return _failure_llm_api_config_payloads()

        cls = llm_args.__class__
        entries = build_capture_manifest(cls)
        state = _CaptureState()
        for entry in entries:
            present, value = _resolve_path(llm_args, entry.path)
            if not present:
                continue
            is_safe, sanitized = _sanitize_value(value, entry.annotation, entry.metadata, state)
            if is_safe:
                state.values[entry.path] = sanitized
            else:
                state.excluded_field_count += 1
                state.unsafe_excluded = True

        config_json = _canonical_json(state.values)
        if len(config_json.encode("utf-8")) > MAX_CONFIG_BYTES:
            state.values, config_json = _truncate_to_budget(state.values)
            state.payload_truncated = True

        rows = manifest_rows(cls)
        metadata = {
            "api_contract_version": API_CONTRACT_VERSION,
            "args_class": cls.__name__,
            "capture_manifest_digest": _digest({"args_class": cls.__name__, "fields": rows}),
            "capture_succeeded": True,
            "capture_version": CAPTURE_VERSION,
            "capturable_field_count": len(entries),
            "captured_field_count": len(state.values),
            "excluded_field_count": state.excluded_field_count,
            "field_policy_version": FIELD_POLICY_VERSION,
            "payload_truncated": state.payload_truncated,
            "schema_digest": _schema_digest(cls),
            "sequence_truncated": state.sequence_truncated,
            "source": CAPTURE_SOURCE,
            "unsafe_excluded": state.unsafe_excluded,
        }
        return config_json, _canonical_json(metadata)
    except (AttributeError, TypeError, ValueError, KeyError):
        # Stay fail-silent only for the sanitizer/walk error family we expect.
        # Unexpected exceptions propagate to the daemon-thread guard in
        # usage_lib so genuine collector bugs are not silently masked.
        args_class = type(llm_args).__name__ if llm_args is not None else ""
        return _failure_llm_api_config_payloads(args_class=args_class)
