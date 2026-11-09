# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LLM API / ``trtllm-serve --config`` fields, read off the checkout's generated JSON Schema.

``tensorrt_llm/schemas/trtllm-serve-config.schema.json`` is generated from the
Pydantic ``TorchLlmArgs`` and committed with every field change, so it is the
checkout's own statement of every field: its dotted path, type, default,
status (``beta``, ``prototype``) and description. Nested configs are
flattened to dotted paths; a discriminated union (``speculative_config``)
contributes the fields of every member, each tagged with the members that
declare it.
"""

from __future__ import annotations

import ast
import difflib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

SERVE_SCHEMA_REL = "tensorrt_llm/schemas/trtllm-serve-config.schema.json"
LLM_ARGS_REL = "tensorrt_llm/llmapi/llm_args.py"
#: Fields ``TorchLlmArgs`` dropped but still takes (and checks) before validation.
_REMOVED_ARGS_NAME = "_TORCH_LLMARGS_REMOVED_ARG_CHECKS"
_MAX_DEPTH = 4


@dataclass
class ConfigField:
    """One LLM API field at its dotted path."""

    path: str
    type: str
    default: Any = None
    has_default: bool = False
    status: str = ""
    description: str = ""
    choices: list[Any] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return schema.get("$defs", {}).get(ref.rsplit("/", 1)[1], {})
    return node


def _members(schema: dict[str, Any], node: dict[str, Any]) -> list[dict[str, Any]]:
    """The non-null alternatives of ``node`` (itself when it has none)."""
    alternatives = node.get("anyOf") or node.get("oneOf")
    if not alternatives:
        return [node]
    found: list[dict[str, Any]] = []
    for alternative in alternatives:
        if alternative.get("type") == "null":
            continue
        if "oneOf" in alternative or "anyOf" in alternative:
            found.extend(_members(schema, alternative))
        else:
            found.append(alternative)
    return found


def _type_name(schema: dict[str, Any], node: dict[str, Any]) -> str:
    names: list[str] = []
    for alternative in node.get("anyOf") or node.get("oneOf") or [node]:
        ref = alternative.get("$ref")
        if isinstance(ref, str):
            names.append(ref.rsplit("/", 1)[1])
        elif "enum" in alternative:
            names.append("enum")
        elif "const" in alternative:
            names.append(repr(alternative["const"]))
        elif alternative.get("type") == "array":
            item = alternative.get("items") or {}
            names.append(f"list[{_type_name(schema, item) if item else 'any'}]")
        elif "type" in alternative:
            kind = alternative["type"]
            names.append("|".join(kind) if isinstance(kind, list) else str(kind))
        elif "oneOf" in alternative or "anyOf" in alternative:
            names.append(_type_name(schema, alternative))
    return "|".join(dict.fromkeys(names)) or "any"


def _choices(schema: dict[str, Any], node: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for member in _members(schema, node):
        target = _resolve(schema, member)
        values.extend(target.get("enum") or [])
        if "const" in target:
            values.append(target["const"])
    return list(dict.fromkeys(values))


def _discriminated(node: dict[str, Any]) -> dict[str, str]:
    """``decoding_type`` value -> definition name, for a discriminated union."""
    for alternative in node.get("anyOf") or [node]:
        mapping = (alternative.get("discriminator") or {}).get("mapping")
        if mapping:
            return {str(k): str(v).rsplit("/", 1)[1] for k, v in mapping.items()}
    return {}


def _walk(
    schema: dict[str, Any],
    properties: dict[str, Any],
    prefix: str,
    depth: int,
    variant: str,
    out: dict[str, ConfigField],
) -> None:
    for name, node in properties.items():
        path = f"{prefix}{name}"
        entry = out.get(path)
        if entry is None:
            entry = ConfigField(
                path=path,
                type=_type_name(schema, node),
                default=node.get("default"),
                has_default="default" in node,
                status=str(node.get("status") or ""),
                description=" ".join(str(node.get("description") or "").split()),
                choices=_choices(schema, node),
            )
            out[path] = entry
        elif variant:
            # The same field of another union member: what any member accepts.
            entry.choices = list(dict.fromkeys([*entry.choices, *_choices(schema, node)]))
            kind = _type_name(schema, node)
            if kind not in entry.type.split("|"):
                entry.type = f"{entry.type}|{kind}"
            if node.get("default") != entry.default:
                entry.default, entry.has_default = None, False
        if variant and variant not in entry.variants:
            entry.variants.append(variant)
        if depth >= _MAX_DEPTH:
            continue
        union = _discriminated(node)
        for member in _members(schema, node):
            target = _resolve(schema, member)
            nested = target.get("properties")
            if not nested:
                continue
            ref = str(member.get("$ref") or "").rsplit("/", 1)[-1]
            tag = next((k for k, v in union.items() if v == ref), "") if union else variant
            _walk(schema, nested, f"{path}.", depth + 1, tag, out)


def removed_fields(repo: Path) -> list[str]:
    """Top-level fields ``TorchLlmArgs`` removed but still accepts, read from ``llm_args.py``."""
    try:
        tree = ast.parse((Path(repo) / LLM_ARGS_REL).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        value = getattr(node, "value", None)
        if isinstance(target, ast.Name) and target.id == _REMOVED_ARGS_NAME:
            if isinstance(value, ast.Dict):
                return sorted(
                    k.value
                    for k in value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                )
    return []


def unknown_fields(
    fields: Iterable[Mapping[str, Any]],
    options: Mapping[str, Any],
    prefix: str = "",
    removed: Iterable[str] = (),
) -> list[str]:
    """Dotted paths in ``options`` that ``fields`` (``extract_config_fields``) do not declare.

    A key under a field that declares no fields of its own (a free-form
    mapping such as ``env_overrides``) is that field's value, not a field.
    ``removed`` fields (``removed_fields``) are still accepted, so not unknown.
    """
    known = {f["path"] for f in fields} | set(removed)
    parents = {p.rsplit(".", 1)[0] for p in known if "." in p}

    def walk(mapping: Mapping[str, Any], at: str) -> list[str]:
        unknown: list[str] = []
        for key, value in mapping.items():
            path = f"{at}{key}"
            if path not in known:
                unknown.append(path)
            elif isinstance(value, Mapping) and path in parents:
                unknown.extend(walk(value, f"{path}."))
        return unknown

    return walk(options, prefix) if known else []


def suggest(fields: Iterable[Mapping[str, Any]], path: str) -> str:
    """The declared field closest to ``path``, or ``""``."""
    close = difflib.get_close_matches(path, [f["path"] for f in fields], n=1, cutoff=0.75)
    return close[0] if close else ""


def read_schema(repo: Path) -> dict[str, Any] | None:
    path = Path(repo) / SERVE_SCHEMA_REL
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def extract_config_fields(repo: Path) -> dict[str, Any]:
    """``{"version", "fields", "decoding_types", "removed"}``; no fields without a schema."""
    schema = read_schema(repo)
    if not schema:
        return {"version": "", "fields": [], "decoding_types": {}, "removed": []}
    fields: dict[str, ConfigField] = {}
    _walk(schema, schema.get("properties") or {}, "", 0, "", fields)
    speculative = (schema.get("properties") or {}).get("speculative_config") or {}
    return {
        "version": str(schema.get("title") or ""),
        "fields": [f.as_dict() for f in sorted(fields.values(), key=lambda f: f.path)],
        "decoding_types": _discriminated(speculative),
        "removed": removed_fields(repo),
    }
