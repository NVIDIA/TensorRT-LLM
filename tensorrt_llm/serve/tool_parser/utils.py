# Adapted from https://github.com/sgl-project/sglang/blob/083629c23564e1a64deaa052f1df5c5d914358d8/python/sglang/srt/function_call/qwen25_detector.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import itertools
import json
import re
from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
from typing import Any, Dict, Iterator, Optional
from urllib.parse import unquote

import partial_json_parser
from partial_json_parser.core.options import Allow


def find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def partial_json_loads(input_str: str, flags: Allow) -> tuple[Any, int]:
    """
    Parse incomplete or partial JSON strings commonly encountered during streaming.

    Args:
        input_str (str): The potentially incomplete JSON string to parse.
        flags (Allow): Bitwise flags controlling what types of partial data are allowed.
            Common flags include:
            - Allow.STR: Allow partial strings (e.g., '"hello wo' -> 'hello wo')
            - Allow.OBJ: Allow partial objects (e.g., '{"key":' -> {'key': None})
            - Allow.ARR: Allow partial arrays (e.g., '[1, 2,' -> [1, 2])
            - Allow.ALL: Allow all types of partial data

    Returns:
        Tuple[Any, int]: A tuple containing:
            - parsed_object: The Python object parsed from the JSON
            - consumed_length: Number of characters consumed from input_str
    """
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (JSONDecodeError, IndexError) as e:
        msg = getattr(e, "msg", str(e))
        if "Extra data" in msg or "pop from empty list" in msg:
            start = WHITESPACE.match(input_str, 0).end()
            obj, end = JSONDecoder().raw_decode(input_str, start)
            return obj, end
        raise


def is_complete_json(input_str: str) -> bool:
    try:
        json.loads(input_str)
        return True
    except JSONDecodeError:
        return False


# The most references one type inference follows, so that references forming a
# cycle, or fanning out, still resolve in bounded time.
_MAX_REFS_FOLLOWED = 32

# A JSON pointer token that indexes an array (RFC 6901, section 4).
_ARRAY_INDEX = re.compile(r"0|[1-9][0-9]*")


def _resolve_local_ref(ref: Any, root: Any) -> Any:
    """The subschema of `root` named by a local reference like ``#/$defs/X``.

    The fragment is percent-decoded into a JSON pointer before its tokens are
    split and unescaped (RFC 6901, section 6), so ``#/$defs/My%20Model`` names
    the definition ``My Model``.

    None for a reference into another document, a named anchor, a fragment
    that is not valid percent-encoded UTF-8, or a pointer that names nothing
    in `root`.
    """
    if not isinstance(ref, str) or not isinstance(root, dict):
        return None
    if not ref.startswith("#"):
        return None
    try:
        pointer = unquote(ref[1:], errors="strict")
    except UnicodeDecodeError:
        return None
    if pointer and not pointer.startswith("/"):
        return None
    node = root
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            if token not in node:
                return None
            node = node[token]
        elif isinstance(node, list):
            # An array index is "0" or ASCII digits without a leading zero
            # (RFC 6901, section 4); "-", other digit characters such as "²",
            # and out-of-range indices name nothing.
            if not _ARRAY_INDEX.fullmatch(token):
                return None
            index = int(token)
            if index >= len(node):
                return None
            node = node[index]
        else:
            return None
    return node


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/utils.py
def infer_type_from_json_schema(
        schema: Dict[str, Any],
        root: Optional[Dict[str, Any]] = None,
        _refs_followed: Optional[Iterator[int]] = None) -> Optional[str]:
    """Infer the primary type of a parameter from JSON Schema.

    Supports complex JSON Schema structures including:
    - Direct type field (including type arrays)
    - $ref: a local reference such as ``#/$defs/Name`` or
      ``#/definitions/Name``, resolved against ``root``, the tool's whole
      parameters schema; a reference that does not resolve gives no type
    - anyOf/oneOf: parameter can be any of multiple types
    - enum: parameter must be one of enum values
    - allOf: parameter must satisfy all type definitions
    - properties: inferred as object type
    - items: inferred as array type

    Returns None when the schema gives no type.
    """
    if not isinstance(schema, dict):
        return None
    if _refs_followed is None:
        _refs_followed = itertools.count()

    if "type" in schema:
        type_value = schema["type"]
        if isinstance(type_value, str):
            return type_value
        elif isinstance(type_value, list) and type_value:
            non_null_types = [t for t in type_value if t != "null"]
            if non_null_types:
                return non_null_types[0]
            return "string"

    if "$ref" in schema and next(_refs_followed) < _MAX_REFS_FOLLOWED:
        inferred_type = infer_type_from_json_schema(
            _resolve_local_ref(schema["$ref"], root), root, _refs_followed)
        if inferred_type:
            return inferred_type

    if "anyOf" in schema or "oneOf" in schema:
        schemas = schema.get("anyOf") or schema.get("oneOf")
        types = []
        if isinstance(schemas, list):
            for sub_schema in schemas:
                inferred_type = infer_type_from_json_schema(
                    sub_schema, root, _refs_followed)
                if inferred_type:
                    types.append(inferred_type)
        if types:
            if len(set(types)) == 1:
                return types[0]
            if "string" in types:
                return "string"
            return types[0]

    if "enum" in schema and isinstance(schema["enum"], list):
        if not schema["enum"]:
            return "string"
        enum_types = set()
        for value in schema["enum"]:
            if value is None:
                enum_types.add("null")
            elif isinstance(value, bool):
                enum_types.add("boolean")
            elif isinstance(value, int):
                enum_types.add("integer")
            elif isinstance(value, float):
                enum_types.add("number")
            elif isinstance(value, str):
                enum_types.add("string")
            elif isinstance(value, list):
                enum_types.add("array")
            elif isinstance(value, dict):
                enum_types.add("object")
        non_null_types = enum_types - {"null"}
        if len(non_null_types) == 1:
            return non_null_types.pop()
        if non_null_types == {"integer", "number"}:
            return "number"
        if enum_types == {"null"}:
            return "null"
        return "string"

    if "allOf" in schema and isinstance(schema["allOf"], list):
        for sub_schema in schema["allOf"]:
            inferred_type = infer_type_from_json_schema(sub_schema, root,
                                                        _refs_followed)
            if inferred_type and inferred_type != "string":
                return inferred_type
        return "string"

    if "properties" in schema:
        return "object"

    if "items" in schema:
        return "array"

    return None
