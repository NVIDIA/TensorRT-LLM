# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for the Kimi K3 XTML output format.

K3 emits tool calls as an XTML tag stream built from the special tokens
``<|open|>`` / ``<|close|>`` / ``<|sep|>`` with plain-text tag headers
(authoritative rendering: the checkpoint's ``encoding_k3.py``)::

    <|open|>tools<|sep|>
      <|open|>call tool="NAME" index="1"<|sep|>
        <|open|>argument key="K" type="string|number|boolean|null|object|array"<|sep|>
          VALUE
        <|close|>argument<|sep|>
        ...
      <|close|>call<|sep|>
      ...
    <|close|>tools<|sep|>

Alternatively a call body may carry one raw JSON block::

    <|open|>json type="object"<|sep|>{...}<|close|>json<|sep|>

Attribute values are escaped (``&`` -> ``&amp;``, ``"`` -> ``&quot;``).
``argument`` bodies are raw text for ``type="string"`` and JSON text for
every other type. The section arrives after the response body, so streaming
buffers the section and emits complete calls once ``<|close|>tools<|sep|>``
is seen.
"""

import functools
import json
import os
import re
from typing import Any, Dict, List, Optional

import xgrammar
from xgrammar import structural_tag as st

from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import StreamingParseResult, ToolCallItem, _GetInfoFunc


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


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


def _free_text() -> st.AnyTextFormat:
    """Channel text that cannot open another channel or end the message.

    The enclosing tag's end string closes it; xgrammar cannot also exclude a
    prefix of that end string.
    """
    return st.AnyTextFormat(excludes=[_OPEN, _END_OF_MSG])


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
            document = json.dumps(_standalone_schema(narrowed, root))
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


def _parse_attrs(header: str) -> Dict[str, str]:
    return {key: _unescape_attr(value) for key, value in re.findall(r'(\w+)="([^"]*)"', header)}


class KimiK3ToolParser(BaseToolParser):
    """Detector for the Kimi K3 XTML function-call format."""

    needs_raw_special_tokens = True
    # Forced/named tool_choice has no grammar for XTML (no structural-tag
    # support), so the model output still carries preamble + markup and the
    # serving layer must extract instead of passing raw text through.
    extracts_forced_tool_calls = True

    def __init__(self):
        super().__init__()
        self.bot_token = "<|open|>tools<|sep|>"  # nosec B105
        self.eot_token = "<|close|>tools<|sep|>"  # nosec B105
        # Set once a complete tools section has been emitted. A K3 tools
        # section terminates the message, so anything streamed afterwards is
        # structural framing, not content.
        self._section_done = False
        # Structural leftovers that may trail the tools section when the
        # reasoning parser is not in front of this parser.
        self._trailing_structural = re.compile(
            r"(?:<\|close\|>message<\|sep\|>|<\|end_of_msg\|>)+\s*$"
        )

        # Tag headers run to the next special token. The encoder escapes only
        # ``&`` and ``"`` in attribute values, so a literal ``<`` (or ``>``)
        # can appear inside one; only ``<|`` is impossible without ending the
        # header, so headers match any text that doesn't contain ``<|``.
        attrs_pattern = r"(?:(?!<\|).)*?"
        self._call_open_regex = re.compile(r"<\|open\|>call(?![a-zA-Z])")
        self._call_regex = re.compile(
            r"<\|open\|>call(?P<attrs>" + attrs_pattern + r")<\|sep\|>"
            r"(?P<body>.*?)<\|close\|>call<\|sep\|>",
            re.DOTALL,
        )
        self._argument_regex = re.compile(
            r"<\|open\|>argument(?P<attrs>" + attrs_pattern + r")<\|sep\|>"
            r"(?P<value>.*?)<\|close\|>argument<\|sep\|>",
            re.DOTALL,
        )
        self._json_regex = re.compile(
            r"<\|open\|>json(?P<attrs>" + attrs_pattern + r")<\|sep\|>"
            r"(?P<value>.*?)<\|close\|>json<\|sep\|>",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def supports_structural_tag(self) -> bool:
        # XTML argument bodies are tag-structured text, not JSON — the
        # generic begin/end/trigger structural-tag path does not apply.
        # Strict tools are handled by build_strict_structural_tag_format.
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "kimi_k3 XTML tool calls do not support structural-tag constrained decoding"
        )

    def build_strict_structural_tag_format(self, tools: List[Tool]) -> Dict[str, Any] | None:
        """Xgrammar structural-tag format enforcing well-formed K3 tool calls.

        Any generated tools section is constrained to calls of the declared
        tools; a strict tool with a parameters schema additionally gets its
        arguments constrained to that JSON Schema via the K3 json-block body
        form (the per-argument XTML form has no xgrammar equivalent).
        Non-strict tools keep free-form bodies. The outer triggered_tags
        must keep `at_least_one`/`stop_after_first` False: True would
        forbid the think/response text before the section and the message
        close after it, deadlocking generation.
        """
        if os.getenv("TRTLLM_KIMI_K3_STRICT_TOOL_GRAMMAR", "0") != "1":
            # Experimental, opt-in: under concurrent guided load with
            # production tool schemas, sampling tripped a device-side assert
            # and hard-killed the deployment (KVV schema suite, job 3054205:
            # TensorCompare.cu _assert_async in sampler.update_requests).
            # Root-cause investigation pending; strict tools fall back to
            # the warn-and-continue path meanwhile.
            return None
        if not tools:
            return None
        for tool in tools:
            if "<" in tool.function.name:
                # A literal '<' in an attribute value has no escaped form in
                # the K3 wire format (the checkpoint renderer escapes only
                # '&' and '"'), and the parser's attribute regex stops at
                # '<' — a grammar-forced call with such a name would be
                # dropped. Skip constrained decoding rather than teach the
                # model a dialect the reference renderer never produces.
                logger.warning(
                    f"Tool name {tool.function.name!r} contains '<'; "
                    "skipping the kimi_k3 strict-tool grammar for this request."
                )
                return None
        call_tags: List[Dict[str, Any]] = []
        for tool in tools:
            begin = f'<|open|>call tool="{_escape_attr(tool.function.name)}"'
            if tool.function.strict and tool.function.parameters:
                call_tags.append(
                    {
                        "type": "tag",
                        "begin": begin,
                        "content": {
                            "type": "sequence",
                            "elements": [
                                {
                                    "type": "regex",
                                    "pattern": ' index="[1-9][0-9]{0,2}"',
                                },
                                {
                                    "type": "const_string",
                                    "value": '<|sep|><|open|>json type="object"<|sep|>',
                                },
                                {
                                    "type": "json_schema",
                                    "json_schema": tool.function.parameters,
                                },
                            ],
                        },
                        "end": "<|close|>json<|sep|><|close|>call<|sep|>",
                    }
                )
            else:
                call_tags.append(
                    {
                        "type": "tag",
                        "begin": begin,
                        "content": {"type": "any_text"},
                        "end": "<|close|>call<|sep|>",
                    }
                )
        return {
            "type": "triggered_tags",
            "triggers": [self.bot_token],
            "tags": [
                {
                    "type": "tag",
                    "begin": self.bot_token,
                    "content": {
                        "type": "tags_with_separator",
                        "separator": "",
                        "at_least_one": True,
                        "tags": call_tags,
                    },
                    "end": self.eot_token,
                }
            ],
            "at_least_one": False,
            "stop_after_first": False,
        }

    def build_required_structural_tag(
        self, tools: List[Tool], *, thinking: bool
    ) -> Optional[st.StructuralTag]:
        """Mandatory-call grammar for ``tool_choice="required"``.

        The message is ``[think body <|close|>think<|sep|> <|open|>response<|sep|>]``
        (the prompt already opened the first channel), the response body,
        ``<|close|>response<|sep|>``, a tools section of one or more calls of
        the declared tools, and ``<|close|>message<|sep|>``; the request's stop
        token follows. Call bodies use the per-argument XTML form the chat
        template renders. Returns None for tool names the K3 wire format
        cannot carry.
        """
        if not tools or any("<" in tool.function.name for tool in tools):
            return None
        elements: List[Any] = []
        if thinking:
            elements += [
                st.TagFormat(begin="", content=_free_text(), end=f"{_CLOSE}think{_SEP}"),
                st.ConstStringFormat(value=f"{_OPEN}response{_SEP}"),
            ]
        elements += [
            st.TagFormat(begin="", content=_free_text(), end=f"{_CLOSE}response{_SEP}"),
            st.TagFormat(
                begin=self.bot_token,
                content=st.TagsWithSeparatorFormat(
                    tags=[_call_tag(tool) for tool in tools], separator="", at_least_one=True
                ),
                end=self.eot_token,
            ),
            st.ConstStringFormat(value=f"{_CLOSE}message{_SEP}"),
        ]
        return st.StructuralTag(format=st.SequenceFormat(elements=elements))

    @staticmethod
    def _coerce_value(value: str, value_type: str) -> Any:
        if value_type == "string":
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(
                f"kimi_k3 tool parser: argument declared type={value_type} but "
                "body is not valid JSON; keeping raw text"
            )
            return value

    def _parse_call_arguments(self, body: str) -> str:
        """Reconstruct the OpenAI ``function.arguments`` JSON string from a call body."""
        json_match = self._json_regex.search(body)
        if json_match is not None:
            raw = json_match.group("value").strip()
            try:
                return json.dumps(json.loads(raw), ensure_ascii=False)
            except json.JSONDecodeError:
                logger.warning(
                    "kimi_k3 tool parser: json block is not valid JSON; passing raw text through"
                )
                return raw
        arguments: Dict[str, Any] = {}
        for match in self._argument_regex.finditer(body):
            attrs = _parse_attrs(match.group("attrs"))
            key = attrs.get("key")
            if key is None:
                continue
            arguments[key] = self._coerce_value(match.group("value"), attrs.get("type", "string"))
        return json.dumps(arguments, ensure_ascii=False)

    def _parse_tools_section(self, section: str, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        opened_calls = len(self._call_open_regex.findall(section))
        matched_calls = 0
        for position, match in enumerate(self._call_regex.finditer(section)):
            matched_calls += 1
            attrs = _parse_attrs(match.group("attrs"))
            name = attrs.get("tool")
            if not name:
                logger.warning(
                    f"kimi_k3 tool parser: call without tool attribute: {match.group('attrs')}"
                )
                continue
            if name not in tool_indices:
                logger.warning(f"Model attempted to call undefined function: {name}")
            calls.append(
                ToolCallItem(
                    tool_index=position,
                    name=name,
                    parameters=self._parse_call_arguments(match.group("body")),
                )
            )
        if matched_calls < opened_calls:
            logger.warning(
                f"kimi_k3 tool parser: {opened_calls - matched_calls} of "
                f"{opened_calls} call blocks were malformed or truncated and "
                "could not be parsed"
            )
        return calls

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        bot_idx = text.find(self.bot_token)
        if bot_idx == -1:
            return StreamingParseResult(normal_text=self._trailing_structural.sub("", text))
        normal_text = text[:bot_idx]
        section = text[bot_idx + len(self.bot_token) :]
        eot_idx = section.find(self.eot_token)
        if eot_idx != -1:
            section = section[:eot_idx]
        calls = self._parse_tools_section(section, tools)
        self.prev_tool_call_arr = []
        for call in calls:
            try:
                arguments = json.loads(call.parameters)
            except json.JSONDecodeError:
                arguments = call.parameters
            self.prev_tool_call_arr.append(
                {
                    "name": call.name,
                    "arguments": arguments,
                }
            )
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        if self._section_done:
            # The completed tools section terminated the K3 message; any later
            # text is structural framing (``<|close|>message<|sep|>``,
            # ``<|end_of_msg|>``), never user content. Buffer it so ``finish``
            # strips it — matching ``detect_and_parse`` — instead of emitting
            # protocol tokens as content.
            return StreamingParseResult()
        bot_idx = self._buffer.find(self.bot_token)
        if bot_idx == -1:
            hold = self._ends_with_partial_token(self._buffer, self.bot_token)
            emit_len = len(self._buffer) - hold
            normal_text = self._buffer[:emit_len]
            self._buffer = self._buffer[emit_len:]
            return StreamingParseResult(normal_text=normal_text)

        # Flush any response text preceding the section, then buffer the
        # whole section until it completes: K3 tool calls terminate the
        # message, so latency cost is negligible and complete calls avoid
        # partial-argument reconstruction entirely.
        normal_text = self._buffer[:bot_idx]
        self._buffer = self._buffer[bot_idx:]
        eot_idx = self._buffer.find(self.eot_token)
        if eot_idx == -1:
            return StreamingParseResult(normal_text=normal_text)
        section_end = eot_idx + len(self.eot_token)
        result = self.detect_and_parse(self._buffer[:section_end], tools)
        # The section terminates the message; hold any trailing framing in the
        # buffer for ``finish`` to strip rather than emitting it as content.
        self._buffer = self._buffer[section_end:]
        self._section_done = True
        return StreamingParseResult(
            normal_text=normal_text + result.normal_text, calls=result.calls
        )

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Emit whatever the buffer holds when the stream ends early.

        ``parse_streaming_increment`` buffers the whole tools section until
        ``<|close|>tools<|sep|>``; if generation stops first (length limit,
        cancellation), the buffered content would otherwise be dropped.
        Complete call blocks are salvaged; a call truncated mid-block is
        reported by the malformed-call warning in ``_parse_tools_section``.
        """
        buffer, self._buffer = self._buffer, ""
        if not buffer:
            return StreamingParseResult()
        if self.bot_token not in buffer:
            # The buffer holds either a partial bot_token prefix or the
            # structural residue left after a completed tools section; the
            # stream is over, so it is plain text after stripping any
            # trailing structural tokens (matching detect_and_parse).
            return StreamingParseResult(normal_text=self._trailing_structural.sub("", buffer))
        logger.warning(
            f"kimi_k3 tool parser: stream ended before {self.eot_token}; "
            "parsing the partial tools section"
        )
        return self.detect_and_parse(buffer, tools)
