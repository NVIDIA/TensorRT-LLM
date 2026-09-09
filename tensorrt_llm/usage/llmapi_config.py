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
FIELD_POLICY_VERSION = "3"
API_CONTRACT_VERSION = "0.2.0"
CAPTURE_SOURCE = "effective_validated_llm_args"

# Cap total serialized bytes of llmApiConfigJson. The wire field is unbounded and
# the reporter is fail-silent, so an oversized payload is dropped whole by the
# endpoint; truncate and flag instead. Conservative bound until the endpoint limit
# is confirmed.
MAX_CONFIG_BYTES = 16384

_TELEMETRY_EXTRA_KEY = "telemetry"
_TRTLLM_JSON_SCHEMA_EXTRA_ATTR = "_trtllm_json_schema_extra"

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


def _normalize_metadata(metadata: Any) -> dict[str, Any] | None:
    if metadata is None:
        return None
    if metadata is False:
        return {"exclude": True}
    if metadata is True:
        return {}
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


def _is_explicit_exclude(metadata: dict[str, Any] | None) -> bool:
    return bool(metadata) and metadata.get("exclude") is True


@dataclass(frozen=True)
class _CapturePolicy:
    """Compiled runtime sanitizer policy for one annotated field."""

    policy_type: str
    runtime_type: Any = None
    allowed_values: tuple[Any, ...] = ()
    branches: tuple["_CapturePolicy", ...] = ()


@dataclass(frozen=True)
class _PolicyVariant:
    """Policy for a field as declared by one reachable model arm."""

    owner: type[BaseModel]
    policy: _CapturePolicy


@dataclass(frozen=True)
class _ManifestEntry:
    path: str
    kind: str
    capture_types: tuple[str, ...]
    allowed_values: tuple[Any, ...]
    variants: tuple[_PolicyVariant, ...]


def _is_safe_scalar(value: Any) -> bool:
    if value is None:
        return True
    if type(value) not in {bool, int, float, str}:
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _typed_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _dedupe_typed(values: list[Any]) -> tuple[Any, ...]:
    deduped: list[Any] = []
    for value in values:
        if not any(_typed_equal(value, seen) for seen in deduped):
            deduped.append(value)
    return tuple(deduped)


def _combine_policies(policies: list[_CapturePolicy]) -> _CapturePolicy | None:
    branches: list[_CapturePolicy] = []
    for policy in policies:
        candidates = policy.branches if policy.policy_type == "union" else (policy,)
        for candidate in candidates:
            if candidate not in branches:
                branches.append(candidate)
    if not branches:
        return None
    if len(branches) == 1:
        return branches[0]
    return _CapturePolicy("union", branches=tuple(branches))


def _sequence_element_annotation(annotation: Any) -> tuple[type, Any] | None:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {list, set} and len(args) == 1:
        return origin, args[0]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return origin, args[0]
    return None


def _compile_annotation_policy(annotation: Any) -> _CapturePolicy | None:
    """Compile only capture-safe annotation branches into a sanitizer policy."""
    annotation = _unwrap_annotated(annotation)

    if annotation is Any or annotation in {str, object}:
        return None
    if annotation is _none_type():
        return _CapturePolicy("none")
    if _is_path_annotation(annotation) or _is_callable_annotation(annotation):
        return None
    if _is_literal(annotation):
        values = list(get_args(annotation))
        if not values or not all(_is_safe_scalar(value) for value in values):
            return None
        return _CapturePolicy("literal", allowed_values=_dedupe_typed(values))
    if _is_enum_annotation(annotation):
        return _CapturePolicy("enum", runtime_type=annotation)
    if annotation in {bool, int, float}:
        return _CapturePolicy(annotation.__name__)
    if _is_union(annotation):
        policies: list[_CapturePolicy] = []
        has_none = False
        for branch in get_args(annotation):
            if _unwrap_annotated(branch) is _none_type():
                has_none = True
                continue
            policy = _compile_annotation_policy(branch)
            if policy is not None:
                policies.append(policy)
        # Optional does not make an otherwise unsafe field capturable by itself.
        if not policies:
            return None
        if has_none:
            policies.append(_CapturePolicy("none"))
        return _combine_policies(policies)

    sequence = _sequence_element_annotation(annotation)
    if sequence is not None:
        origin, element_annotation = sequence
        element_policy = _compile_annotation_policy(element_annotation)
        if element_policy is not None:
            return _CapturePolicy("sequence", runtime_type=origin, branches=(element_policy,))
    return None


def _annotation_allows_none(annotation: Any) -> bool:
    annotation = _unwrap_annotated(annotation)
    return annotation is _none_type() or (
        _is_union(annotation)
        and any(_unwrap_annotated(branch) is _none_type() for branch in get_args(annotation))
    )


def _annotation_accepts_allowed_value(annotation: Any, value: Any) -> bool:
    """Whether an unsafe annotation branch can own an explicit scalar token."""
    annotation = _unwrap_annotated(annotation)
    if annotation is Any or annotation is object:
        return _is_safe_scalar(value)
    if annotation is str:
        return type(value) is str
    if _is_path_annotation(annotation) or _is_callable_annotation(annotation):
        return False
    if _is_union(annotation):
        return any(
            _annotation_accepts_allowed_value(branch, value) for branch in get_args(annotation)
        )
    return False


def _explicit_allowlist_policy(annotation: Any, metadata: dict[str, Any]) -> _CapturePolicy | None:
    if "allowed_values" not in metadata:
        return None
    allowed_values = metadata["allowed_values"]
    if not isinstance(allowed_values, (list, tuple, set)) or not allowed_values:
        raise ValueError("telemetry allowed_values must be a non-empty sequence")
    values = list(allowed_values)
    if isinstance(allowed_values, set):
        values.sort(key=lambda value: f"{type(value).__name__}:{_canonical_json(value)}")
    if not all(_is_safe_scalar(value) for value in values):
        raise ValueError("telemetry allowed_values must contain only finite JSON scalars")
    if not all(_annotation_accepts_allowed_value(annotation, value) for value in values):
        raise ValueError(
            "telemetry allowed_values must belong to an unsafe scalar annotation branch"
        )
    return _CapturePolicy("allowlist", allowed_values=_dedupe_typed(values))


def _compile_field_policy(annotation: Any, metadata: dict[str, Any]) -> _CapturePolicy | None:
    annotation_policy = _compile_annotation_policy(annotation)
    policies: list[_CapturePolicy] = []
    allowlist_policy = _explicit_allowlist_policy(annotation, metadata)
    if allowlist_policy is not None:
        policies.append(allowlist_policy)
        if annotation_policy is None and _annotation_allows_none(annotation):
            policies.append(_CapturePolicy("none"))
    if annotation_policy is not None:
        policies.append(annotation_policy)
    return _combine_policies(policies)


def _enum_output(member: Enum) -> str:
    return member.value if isinstance(member.value, str) else member.name


def _policy_allowed_values(policy: _CapturePolicy) -> tuple[Any, ...]:
    values: list[Any] = []
    if policy.policy_type in {"literal", "allowlist"}:
        values.extend(policy.allowed_values)
    elif policy.policy_type == "enum":
        values.extend(_enum_output(member) for member in policy.runtime_type)
    elif policy.policy_type in {"union", "sequence"}:
        for branch in policy.branches:
            values.extend(_policy_allowed_values(branch))
    return _dedupe_typed(values)


def _policy_signature(policy: _CapturePolicy) -> str:
    if policy.policy_type == "sequence":
        return f"{policy.runtime_type.__name__}[{_policy_signature(policy.branches[0])}]"
    if policy.policy_type == "union":
        signatures = sorted({_policy_signature(branch) for branch in policy.branches})
        return "|".join(signatures)
    if policy.policy_type == "enum":
        return f"enum[{policy.runtime_type.__name__}]"
    return policy.policy_type


def _policy_capture_types(policy: _CapturePolicy) -> tuple[str, ...]:
    branches = policy.branches if policy.policy_type == "union" else (policy,)
    return tuple(sorted({_policy_signature(branch) for branch in branches}))


def _policy_kind(policy: _CapturePolicy) -> str:
    if policy.policy_type in {"literal", "enum", "allowlist"}:
        return "categorical"
    if any(_policy_kind(branch) == "categorical" for branch in policy.branches):
        return "categorical"
    return "value"


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


def build_capture_manifest(model_cls: type[BaseModel]) -> list[_ManifestEntry]:
    """Walk real type objects and emit the complete capturable manifest.

    The single source of truth. Type-safe annotations auto-enroll; str/Any
    allowlist escape hatches opt in; telemetry=False opts out. Recurses into
    statically reachable nested BaseModels with a cycle guard. Collapses
    duplicate keys shared by nested model union arms while retaining an
    owner-specific compiled policy for every arm. Display domains and capture
    types are merged, but runtime sanitization selects only the active arm's
    policy. Conflicting kinds fail manifest construction.
    """
    rows: list[dict[str, Any]] = []

    def walk(cls: type, prefix: str, stack: tuple) -> None:
        if cls in stack:
            return
        for fname, finfo in cls.model_fields.items():
            key = f"{prefix}.{fname}" if prefix else fname
            ann = finfo.annotation
            meta = _get_telemetry_metadata(finfo)
            normalized = meta if (meta and not _is_explicit_exclude(meta)) else {}
            policy = None if _is_explicit_exclude(meta) else _compile_field_policy(ann, normalized)
            if policy is not None:
                rows.append(
                    {
                        "key": key,
                        "defining": _defining_class(cls, fname),
                        "owner": cls,
                        "policy": policy,
                        "kind": _policy_kind(policy),
                    }
                )
            if not _is_explicit_exclude(meta):
                for sub in _nested_models(ann):
                    walk(sub, key, (*stack, cls))

    walk(model_cls, "", ())

    rows.sort(key=lambda r: (r["key"], r["defining"], r["owner"].__qualname__))
    grouped: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = r["key"]
        if key not in grouped:
            grouped[key] = {"kind": r["kind"], "variants": []}
        elif grouped[key]["kind"] != r["kind"]:
            raise ValueError(
                f"telemetry manifest: key '{key}' has conflicting kinds "
                f"across model arms: {grouped[key]['kind']} vs {r['kind']}"
            )
        variants: list[_PolicyVariant] = grouped[key]["variants"]
        matching = [variant for variant in variants if variant.owner is r["owner"]]
        if matching:
            if any(variant.policy != r["policy"] for variant in matching):
                raise ValueError(
                    f"telemetry manifest: key '{key}' has conflicting policies "
                    f"for model arm {r['owner'].__qualname__}"
                )
            continue
        variants.append(_PolicyVariant(owner=r["owner"], policy=r["policy"]))

    entries = []
    for key, group in grouped.items():
        variants = tuple(group["variants"])
        allowed_values: list[Any] = []
        capture_types: set[str] = set()
        for variant in variants:
            allowed_values.extend(_policy_allowed_values(variant.policy))
            capture_types.update(_policy_capture_types(variant.policy))
        entries.append(
            _ManifestEntry(
                path=key,
                kind=group["kind"],
                capture_types=tuple(sorted(capture_types)),
                allowed_values=_dedupe_typed(allowed_values),
                variants=variants,
            )
        )
    entries.sort(key=lambda e: e.path)
    return entries


def manifest_rows(model_cls: type[BaseModel]) -> list[dict[str, Any]]:
    """Serializable, human-legible projection of build_capture_manifest.

    Used by the committed golden, the docs renderer, and the
    capture_manifest_digest.
    """
    rows = []
    for entry in build_capture_manifest(model_cls):
        row = {
            "path": entry.path,
            "kind": entry.kind,
            "capture_policy": "|".join(entry.capture_types),
        }
        if entry.allowed_values:
            row["allowed_values"] = list(entry.allowed_values)
        rows.append(row)
    return rows


def golden_manifest() -> dict[str, list[dict[str, Any]]]:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    return {
        "TorchLlmArgs": manifest_rows(TorchLlmArgs),
    }


def _sanitize_allowed_value(value: Any, allowed_values: tuple[Any, ...]) -> tuple[bool, Any]:
    """Capture only a finite scalar matching an exact typed allowed value."""
    candidates = [value]
    if isinstance(value, Enum):
        enum_output = _enum_output(value)
        candidates.extend((value.value, enum_output))
        if isinstance(enum_output, str):
            candidates.append(enum_output.lower())
    for candidate in candidates:
        if _is_safe_scalar(candidate) and any(
            _typed_equal(candidate, allowed) for allowed in allowed_values
        ):
            return True, candidate
    return False, None


def _sanitize_policy(
    value: Any,
    policy: _CapturePolicy,
    state: _CaptureState | None = None,
) -> tuple[bool, Any]:
    """Try the compiled policy branches and return one telemetry-safe value."""
    if policy.policy_type == "union":
        for branch in policy.branches:
            is_safe, sanitized = _sanitize_policy(value, branch, state)
            if is_safe:
                return True, sanitized
        return False, None
    if policy.policy_type == "none":
        return (True, None) if value is None else (False, None)
    if policy.policy_type == "bool":
        return (True, value) if type(value) is bool else (False, None)
    if policy.policy_type == "int":
        return (True, value) if type(value) is int else (False, None)
    if policy.policy_type == "float":
        # The Python numeric tower permits an int where float is annotated,
        # and Pydantic does not validate every default. Normalize such values
        # to the annotation's JSON number shape, while still excluding bool.
        if type(value) not in {int, float}:
            return False, None
        try:
            normalized = float(value)
        except OverflowError:
            return False, None
        return (True, normalized) if math.isfinite(normalized) else (False, None)
    if policy.policy_type in {"literal", "allowlist"}:
        return _sanitize_allowed_value(value, policy.allowed_values)
    if policy.policy_type == "enum":
        if type(value) is not policy.runtime_type:
            return False, None
        return True, _enum_output(value)
    if policy.policy_type == "sequence":
        if type(value) is not policy.runtime_type:
            return False, None
        element_policy = policy.branches[0]
        sanitized = []
        for item in value:
            item_safe, item_value = _sanitize_policy(item, element_policy, state)
            if not item_safe:
                return False, None
            sanitized.append(item_value)
        if policy.runtime_type is set:
            sanitized.sort(key=_canonical_json)
        if len(sanitized) > MAX_SEQ_ITEMS:
            sanitized = sanitized[:MAX_SEQ_ITEMS]
            if state is not None:
                state.sequence_truncated = True
        return True, sanitized
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


def _resolve_path(instance: BaseModel, path: str) -> tuple[bool, Any, BaseModel | None]:
    """Resolve a dotted manifest path against a live instance.

    Returns (present, value, leaf owner). Skips when a parent segment is
    missing/None or is not a pydantic model (unset config, or a union arm that
    is not active). The owner selects that arm's compiled capture policy.
    """
    segments = path.split(".")
    obj: Any = instance
    for seg in segments[:-1]:
        if not _is_pydantic_model(obj):
            return False, None, None
        if seg not in obj.__class__.model_fields:
            return False, None, None
        obj = getattr(obj, seg, None)
        if obj is None:
            return False, None, None
    leaf = segments[-1]
    if not _is_pydantic_model(obj) or leaf not in obj.__class__.model_fields:
        return False, None, None
    return True, getattr(obj, leaf, None), obj


def _policy_for_owner(entry: _ManifestEntry, owner: BaseModel) -> _CapturePolicy | None:
    exact = [variant.policy for variant in entry.variants if type(owner) is variant.owner]
    if exact:
        return _combine_policies(exact)

    owner_mro = type(owner).__mro__
    compatible = [
        (owner_mro.index(variant.owner), variant.policy)
        for variant in entry.variants
        if variant.owner in owner_mro
    ]
    if not compatible:
        return None
    closest_distance = min(distance for distance, _ in compatible)
    closest = [policy for distance, policy in compatible if distance == closest_distance]
    if any(policy != closest[0] for policy in closest[1:]):
        raise ValueError(
            f"telemetry manifest: key '{entry.path}' has ambiguous policies "
            f"for active model arm {type(owner).__qualname__}"
        )
    return closest[0]


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
            present, value, owner = _resolve_path(llm_args, entry.path)
            if not present or owner is None:
                continue
            policy = _policy_for_owner(entry, owner)
            if policy is None:
                continue
            is_safe, sanitized = _sanitize_policy(value, policy, state)
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
