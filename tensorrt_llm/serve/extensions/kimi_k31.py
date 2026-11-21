# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 serving extension and required-tool grammar."""

from __future__ import annotations

import functools
import json
from typing import Any, Dict, List, Optional

import xgrammar
from xgrammar import structural_tag as st

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.extensions.kimi_k3 import KimiK3ServingExtension
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.serving_extensions import RequiredToolChoiceMode, register_serving_extension
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.kimi_k3_tool_parser import KimiK3ToolParser


def _escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


_OPEN = "<|open|>"
_CLOSE = "<|close|>"
_SEP = "<|sep|>"
_END_OF_MSG = "<|end_of_msg|>"
_ARGUMENT_CLOSE = f"{_CLOSE}argument{_SEP}"

# JSON Schema type -> XTML ``type`` attribute (encoding_k3._xtml_type folds
# integer into number).
_XTML_TYPES = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
    "object": "object",
    "array": "array",
}

# One raw string-value character: anything but the "<|" that starts a marker.
_RAW_STRING_CHAR = r"(?:[^<]|<[^|])"


# The K3 message-level tags whose close a channel's text must not contain.
_MESSAGE_TAGS = ("think", "response", "tools", "message")


def _free_text(channel: str) -> st.AnyTextFormat:
    """Text of one channel: no tag opens, no message end, no close of another message-level tag.

    The enclosing tag's end string (``<|close|>CHANNEL<|sep|>``) closes it; xgrammar cannot
    also exclude a prefix of that end string, so the other closes are excluded by name. Without
    them a response that closes as ``<|close|>think<|sep|>`` (or ``<|close|>message<|sep|>``)
    stays inside the free text, where the reopening ``<|open|>`` and the stop ``<|end_of_msg|>``
    are masked, and generation repeats ``<|close|>message<|sep|>`` until the token limit.
    """
    others = [f"{_CLOSE}{tag}" for tag in _MESSAGE_TAGS if tag != channel]
    return st.AnyTextFormat(excludes=[_OPEN, _END_OF_MSG, *others])


def _any_argument_tag() -> st.TagFormat:
    """An ``argument`` tag with any header and a raw value."""
    return st.TagFormat(
        begin=f"{_OPEN}argument ",
        content=st.SequenceFormat(
            elements=[
                st.RegexFormat(pattern=r"[^<]*<\|sep\|>"),
                st.AnyTextFormat(),
            ]
        ),
        end=_ARGUMENT_CLOSE,
    )


_DEFS_KEYS = ("$defs", "definitions")
# The ``$defs`` entry that carries the whole parameters document into a
# property's standalone schema, for pointers such as ``#`` that address it.
_PARAMETERS_DEF = "__kimi_k3_parameters"
# Keywords that only describe a schema; a property with nothing else accepts any value.
_ANNOTATION_KEYWORDS = frozenset(
    {
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "$comment",
    }
)


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "object" if isinstance(value, dict) else "array"


def _is_local_pointer(ref: str) -> bool:
    return ref == "#" or ref.startswith("#/")


def _resolve_pointer(root: Dict[str, Any], ref: str) -> Any:
    """The node a local JSON pointer (``#`` or ``#/a/b``) addresses in ``root``, or None."""
    node: Any = root
    for part in ref[2:].split("/") if ref != "#" else []:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def _refs(node: Any):
    if isinstance(node, dict):
        if isinstance(node.get("$ref"), str):
            yield node["$ref"]
        for value in node.values():
            yield from _refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _refs(value)


def _value_types(schema: Any, root: Dict[str, Any], seen: tuple = ()) -> Optional[set]:
    """The JSON types a value valid against ``schema`` can have; None when any type can."""
    if not isinstance(schema, dict):
        return None
    json_type = schema.get("type")
    if isinstance(json_type, str):
        return {json_type}
    if isinstance(json_type, list) and json_type and all(isinstance(t, str) for t in json_type):
        return set(json_type)
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return {_json_type(v) for v in schema["enum"]}
    if "const" in schema:
        return {_json_type(schema["const"])}
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen or not _is_local_pointer(ref):
            return None
        return _value_types(_resolve_pointer(root, ref), root, (*seen, ref))
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and branches:
            types: set = set()
            for branch in branches:
                branch_types = _value_types(branch, root, seen)
                if branch_types is None:
                    return None
                types |= branch_types
            return types
    narrowed: Optional[set] = None
    for branch in schema.get("allOf") or []:
        branch_types = _value_types(branch, root, seen)
        if branch_types is not None:
            narrowed = branch_types if narrowed is None else narrowed & branch_types
    return narrowed or None


def _string_schema(schema: Any, root: Dict[str, Any]) -> Dict[str, Any]:
    """The string schema whose enum or length bounds a raw string value must meet, or {}."""
    seen = set()
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str):
        ref = schema["$ref"]
        if ref in seen or not _is_local_pointer(ref):
            return {}
        seen.add(ref)
        schema = _resolve_pointer(root, ref)
    if not isinstance(schema, dict):
        return {}
    if schema.get("type") == "string":
        return schema
    branches = [
        b
        for key in ("anyOf", "oneOf")
        for b in schema.get(key) or []
        if _value_types(b, root) == {"string"}
    ]
    return _string_schema(branches[0], root) if len(branches) == 1 else {}


def _standalone_schema(schema: Dict[str, Any], root: Dict[str, Any]) -> Dict[str, Any]:
    """``schema``, cut out of the parameters document ``root``, as a standalone document.

    The document's definition tables move along to the new root, so
    ``#/$defs/...`` and ``#/definitions/...`` pointers keep resolving. Any other
    local pointer (``#`` itself, ``#/properties/...``) addresses the parameters
    document, which is carried as one more ``$defs`` entry.
    """
    uses_root = False

    def rewrite(node: Any) -> Any:
        nonlocal uses_root
        if isinstance(node, list):
            return [rewrite(value) for value in node]
        if not isinstance(node, dict):
            return node
        out = {key: rewrite(value) for key, value in node.items()}
        ref = node.get("$ref")
        if isinstance(ref, str) and _is_local_pointer(ref):
            if not ref.startswith(tuple(f"#/{key}/" for key in _DEFS_KEYS)):
                uses_root = True
                out["$ref"] = f"#/$defs/{_PARAMETERS_DEF}{ref[1:]}"
        return out

    document = rewrite(schema)
    tables = {key: rewrite(root[key]) for key in _DEFS_KEYS if isinstance(root.get(key), dict)}
    needs_parameters = uses_root
    parameters = rewrite({key: value for key, value in root.items() if key not in _DEFS_KEYS})
    for key, table in tables.items():
        own = document.get(key)
        document[key] = {**own, **table} if isinstance(own, dict) else table
    if needs_parameters:
        document["$defs"] = {**document.get("$defs", {}), _PARAMETERS_DEF: parameters}
    return document


# JSON-schema keywords whose values are subschemas: a map of them, one, or a list.
_SUBSCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SUBSCHEMA_ONE = (
    "items",
    "additionalProperties",
    "additionalItems",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
)
_SUBSCHEMA_LISTS = ("items", "prefixItems", "anyOf", "oneOf", "allOf")


def _without_negative_lower_bounds(schema: Any) -> Any:
    """``schema`` without a negative ``minimum`` / ``exclusiveMinimum`` in it or its subschemas.

    xgrammar's range regex for a negative lower bound admits a bare "-" and rejects in-range negative numbers;
    without the bound the grammar still enforces the type. Only schema positions are rewritten: the values of other
    keywords (``enum``, ``const``, ``default``, ``examples``, ...) are kept as they are.
    """
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if (
            key in ("minimum", "exclusiveMinimum")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value < 0
        ):
            continue
        if key in _SUBSCHEMA_MAPS and isinstance(value, dict):
            value = {name: _without_negative_lower_bounds(sub) for name, sub in value.items()}
        elif key in _SUBSCHEMA_LISTS and isinstance(value, list):
            value = [_without_negative_lower_bounds(sub) for sub in value]
        elif key in _SUBSCHEMA_ONE and isinstance(value, dict):
            value = _without_negative_lower_bounds(value)
        out[key] = value
    return out


@functools.lru_cache(maxsize=1024)
def _json_schema_compiles(document: str) -> bool:
    try:
        xgrammar.Grammar.from_json_schema(document)
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning(f"kimi_k3 required-tool grammar: keeping a parameter unconstrained, {e}")
        return False
    return True


def _raw_string_content(schema: Dict[str, Any]) -> Any:
    """Raw string values: only an enum or a length bound can constrain them."""
    values = schema.get("enum")
    if values is None and isinstance(schema.get("const"), str):
        values = [schema["const"]]
    max_len = schema.get("maxLength")
    if (
        isinstance(values, list)
        and values
        and all(isinstance(v, str) and "<|" not in v for v in values)
    ):
        options = [st.ConstStringFormat(value=v) for v in values]
        return options[0] if len(options) == 1 else st.OrFormat(elements=options)
    if isinstance(max_len, int) and 0 <= max_len <= 4096:
        min_len = schema.get("minLength", 0)
        if not isinstance(min_len, int) or not 0 <= min_len <= max_len:
            min_len = 0
        return st.RegexFormat(pattern=f"{_RAW_STRING_CHAR}{{{min_len},{max_len}}}")
    return st.AnyTextFormat()


def _narrowed_schema(
    schema: Dict[str, Any], root: Dict[str, Any], xtml_type: str
) -> Dict[str, Any]:
    """``schema`` cut down to the values the K3 renderer writes with ``xtml_type``.

    The tag's ``type`` attribute then matches its value. xgrammar 0.1.32 ignores
    ``allOf`` and a sibling ``type`` next to ``anyOf``, so the union's branches
    of other types are dropped instead.
    """
    json_types = {t for t, x in _XTML_TYPES.items() if x == xtml_type}
    seen = set()
    while (
        set(schema) - _ANNOTATION_KEYWORDS == {"$ref"}
        and isinstance(schema["$ref"], str)
        and schema["$ref"] not in seen
    ):
        seen.add(schema["$ref"])
        target = _resolve_pointer(root, schema["$ref"])
        if not isinstance(target, dict):
            break
        schema = target
    out = dict(schema)
    if isinstance(out.get("type"), list):
        out["type"] = [t for t in out["type"] if t in json_types]
    for key in ("anyOf", "oneOf"):
        if isinstance(out.get(key), list):
            out[key] = [
                branch
                for branch in out[key]
                if (types := _value_types(branch, root)) is None or types & json_types
            ]
    return out


def _argument_tags(key: str, schema: Any, root: Dict[str, Any]) -> List[st.TagFormat]:
    """The ``argument`` tags one parameter can be written with, one per XTML type.

    The type follows the value as the K3 renderer writes it (``type="null"`` for
    null, ``type="object"`` for an object, ...), so a parameter whose schema
    admits several types (``$ref``, ``anyOf``, a type list) gets one tag per
    type. Strings are written raw; every other type is written as JSON that must
    match the parameter's schema restricted to that type. Only a parameter that
    constrains nothing, or whose schema xgrammar cannot compile, keeps a
    permissive tag.
    """
    if (
        not isinstance(schema, dict)
        or set(schema) <= _ANNOTATION_KEYWORDS
        or not all(_is_local_pointer(ref) for ref in _refs(schema))
    ):
        return [_any_argument_tag()]
    json_types = _value_types(schema, root)
    if json_types is not None and not json_types <= _XTML_TYPES.keys():
        return [_any_argument_tag()]
    tags = []
    for xtml_type in sorted({_XTML_TYPES[t] for t in json_types or _XTML_TYPES}):
        if xtml_type == "string":
            content: Any = _raw_string_content(_string_schema(schema, root))
        else:
            narrowed = _narrowed_schema(schema, root, xtml_type) if json_types else schema
            document = json.dumps(
                _without_negative_lower_bounds(_standalone_schema(narrowed, root))
            )
            if not _json_schema_compiles(document):
                return [_any_argument_tag()]
            content = st.JSONSchemaFormat(json_schema=json.loads(document))
        tags.append(
            st.TagFormat(
                begin=f'{_OPEN}argument key="{_escape_attr(key)}" type="{xtml_type}"{_SEP}',
                content=content,
                end=_ARGUMENT_CLOSE,
            )
        )
    return tags


def _call_tag(tool: Tool) -> st.TagFormat:
    """``<|open|>call tool="NAME" index="N"<|sep|>`` and the call's arguments."""
    parameters = tool.function.parameters
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    if isinstance(properties, dict) and properties:
        argument_tags = [
            tag for key, prop in properties.items() for tag in _argument_tags(key, prop, parameters)
        ]
        required = parameters.get("required")
        at_least_one = isinstance(required, list) and bool(required)
    else:
        argument_tags = [_any_argument_tag()]
        at_least_one = False
    return st.TagFormat(
        begin=f'{_OPEN}call tool="{_escape_attr(tool.function.name)}" index="',
        content=st.SequenceFormat(
            elements=[
                st.RegexFormat(pattern="[0-9]+"),
                st.ConstStringFormat(value=f'"{_SEP}'),
                st.TagsWithSeparatorFormat(
                    tags=argument_tags, separator="", at_least_one=at_least_one
                ),
            ]
        ),
        end=f"{_CLOSE}call{_SEP}",
    )


@register_serving_extension(model_types=("kimi_k31",), reasoning_parsers=("kimi_k31",))
class K31ServingExtension(KimiK3ServingExtension):
    """Kimi K3's serving semantics for kimi_k31 chat requests, with a grammar for required tool calls."""

    def required_tool_choice_mode(self) -> RequiredToolChoiceMode:
        """Enforce the call with the kimi_k3 tool grammar when the request allows it."""
        return "prefer_grammar"

    def build_required_structural_tag(
        self, parser: BaseToolParser, tools: List[Tool], *, thinking: bool
    ) -> Optional[st.StructuralTag]:
        """Mandatory-call grammar for ``tool_choice="required"``.

        The message is ``[think body <|close|>think<|sep|> <|open|>response<|sep|>]``
        (the prompt already opened the first channel), the response body,
        ``<|close|>response<|sep|>``, a tools section of one or more calls of
        the declared tools, and ``<|close|>message<|sep|>``; the request's stop
        token follows. Call bodies use the per-argument XTML form the chat
        template renders. Returns None for tool names the K3 wire format cannot
        carry. A server configured with another tool parser gets that parser's
        grammar.
        """
        if not isinstance(parser, KimiK3ToolParser):
            return super().build_required_structural_tag(parser, tools, thinking=thinking)
        if not tools or any("<" in tool.function.name for tool in tools):
            return None

        elements: List[Any] = []
        if thinking:
            elements += [
                st.TagFormat(begin="", content=_free_text("think"), end=f"{_CLOSE}think{_SEP}"),
                st.ConstStringFormat(value=f"{_OPEN}response{_SEP}"),
            ]
        elements += [
            st.TagFormat(begin="", content=_free_text("response"), end=f"{_CLOSE}response{_SEP}"),
            st.TagFormat(
                begin=parser.bot_token,
                content=st.TagsWithSeparatorFormat(
                    tags=[_call_tag(tool) for tool in tools], separator="", at_least_one=True
                ),
                end=parser.eot_token,
            ),
            st.ConstStringFormat(value=f"{_CLOSE}message{_SEP}"),
        ]
        return st.StructuralTag(format=st.SequenceFormat(elements=elements))
