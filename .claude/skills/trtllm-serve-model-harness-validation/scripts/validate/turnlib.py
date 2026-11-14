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
"""The turn record (references/turn-ir.md) and the protocol-independent call rules.

Every protocol adapter (``proto_*.py``) turns its wire format into the classes
below; every check reads only them. Content (texts, payloads) lives in memory
while one reply is checked and never reaches a report: reports carry lengths,
SHA-256 digests and locators.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import jsonschema
except ImportError:  # payloads are then not schema-checked, and that is reported
    jsonschema = None
try:
    from referencing.exceptions import Unresolvable  # jsonschema 4.18+: a $ref it cannot resolve
except ImportError:
    Unresolvable = None
try:
    import lark
except ImportError:
    lark = None

# A declared schema no payload can be checked against: invalid, or a $ref that does not resolve.
SCHEMA_UNUSABLE = tuple(e for e in (getattr(jsonschema, "SchemaError", None), Unresolvable) if e)

NESTED_HEADER = re.compile(r"^### `([A-Za-z0-9_.-]+)`", re.M)
QUALIFIED_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)+$")
QUOTES = "`'\""

SERVING_SHAPE = ("type_mismatch", "namespace_mismatch", "prefixed_name_of_declared")
MODEL_CLASSES = ("undeclared", "nested_tool_called_directly", "payload_invalid")


# --------------------------------------------------------------------------- records


@dataclass
class Tool:
    name: str
    namespace: Optional[str] = None
    kind: str = "function"  # function | custom | builtin | server
    declared_in: str = "tools"
    schema: Optional[dict] = None
    grammar: Optional[dict] = None
    description: str = ""


@dataclass
class Item:
    kind: str  # text | reasoning | call
    index: int = 0
    id: Optional[str] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    namespace: Optional[str] = None
    call_kind: Optional[str] = None  # function | custom
    payload: Optional[str] = None
    text: Optional[str] = None
    carrier: Optional[str] = None  # present | absent
    status: Optional[str] = None
    carrier_digest: Optional[str] = None  # sha256 of the carrier's value, when the protocol has one


@dataclass
class Violation:
    rule: str  # protocols.md rule id, e.g. ANT-06
    detail: dict = field(default_factory=dict)


@dataclass
class RequestView:
    protocol: str
    route: str
    stream: bool = False
    model_field: Optional[str] = None
    tools: Dict[str, Tool] = field(default_factory=dict)
    nested: Dict[str, str] = field(default_factory=dict)  # nested name -> declaring tool
    tools_key: Optional[str] = None
    # Every tool declared, as USG-08 weighs a change of tools: tool_declarations().
    declarations: tuple = ()
    tool_choice: dict = field(default_factory=lambda: {"mode": "auto", "name": None})
    parallel: str = "unspecified"  # allowed | single | unspecified
    reasoning: dict = field(default_factory=dict)
    max_output_tokens: Optional[int] = None
    include_usage: Optional[bool] = None
    choices: int = 1  # Chat Completions `n`
    unknown_fields: List[str] = field(default_factory=list)
    # Our earlier items as the harness sent them back, by id or call id.
    echo: Dict[str, dict] = field(default_factory=dict)
    # (call_id, result text, is_error) for every tool result in the history.
    tool_results: List[Tuple[str, str, Optional[bool]]] = field(default_factory=list)
    history_messages: int = 0
    # Messages and Chat: (position, digest of the message before it, text and reasoning blocks)
    # for every assistant message, so a turn's text echo is found where it was appended.
    assistant_turns: List[tuple] = field(default_factory=list)
    last_message_digest: Optional[str] = None


@dataclass
class ReplyView:
    kind: str = "none"  # sse | json | text | none
    http_status: Optional[int] = None
    status: Optional[str] = None  # the trace/replay record's transport status
    events: int = 0
    grammar: List[Violation] = field(default_factory=list)
    stream_items: List[Item] = field(default_factory=list)
    final_items: Optional[List[Item]] = None
    final_na_reason: Optional[str] = None
    terminal: dict = field(default_factory=lambda: {"kind": "none"})
    usage: dict = field(default_factory=dict)
    first_event_s: Optional[float] = None
    max_gap_s: Optional[float] = None
    timing_known: bool = False
    extras: dict = field(default_factory=dict)

    def delivered(self) -> List[Item]:
        """The items the client acted on: the stream when streamed, else the final body."""
        return self.stream_items if self.kind == "sse" else (self.final_items or [])


# --------------------------------------------------------------------------- JSON


def strict_json(text: str) -> Any:
    """json.loads that rejects duplicate keys and non-finite numbers."""

    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return dict(items)

    def reject_constant(name):
        raise ValueError("non-finite number " + name)

    def finite_float(literal):
        value = float(literal)
        if not math.isfinite(value):
            raise ValueError("number overflows to non-finite")
        return value

    return json.loads(
        text, object_pairs_hook=pairs, parse_constant=reject_constant, parse_float=finite_float
    )


def json_object(text: str) -> Optional[dict]:
    """``text`` read as a JSON object (blank reads as ``{}``), or None when it is not one.

    JSON's grammar has no NaN or Infinity, which Python's json.loads takes and JavaScript's
    JSON.parse rejects: a text that holds them is not JSON. A repeated key is JSON's grammar.
    """
    try:
        value = json.loads(text if text.strip() else "{}", parse_constant=_no_constant)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _canonical(value: Any) -> Any:
    """A form in which equal JSON values compare equal and types never mix.

    Booleans never equal numbers (Python's ``True == 1`` would), key order is
    irrelevant, list order matters, and 1 equals 1.0. Integers stay exact: no float
    holds every integer, and some JSON integers are too large for any float.
    """
    if isinstance(value, bool):
        return ["b", value]
    if isinstance(value, float) and value.is_integer():
        return ["n", int(value)]
    if isinstance(value, (int, float)):
        return ["n", value]
    if isinstance(value, str):
        return ["s", value]
    if value is None:
        return ["z"]
    if isinstance(value, list):
        return ["l", [_canonical(v) for v in value]]
    if isinstance(value, dict):
        return ["o", sorted([k, _canonical(v)] for k, v in value.items())]
    return ["?", repr(value)]


def payload_digest(payload: Optional[str], call_kind: Optional[str]) -> Optional[str]:
    """Digest of a call payload: typed JSON for function arguments, exact bytes otherwise."""
    if payload is None:
        return None
    if call_kind == "function":
        try:
            value = strict_json(payload) if payload.strip() else {}
        except ValueError:
            return "raw:" + sha256_text(payload)
        return "json:" + sha256_text(json.dumps(_canonical(value), separators=(",", ":")))
    return "raw:" + sha256_text(payload)


# A leading ``cd <dir> &&`` in a shell command (Claude Code drops it when <dir> is its cwd).
CD_PREFIX = re.compile(r"""cd\s+(?:"[^"]*"|'[^']*'|[^\s"';&|]+)\s*&&\s*""")
ECHO_REWRITES = ("schema_defaults", "cd_cwd_prefix", "pi_edit_normalize", "pi_json_reserialize")
# Rewrites of what we delivered only: the harness made them, so an echo is never rewritten back.
DELIVERED_ONLY_REWRITES = ("cd_cwd_prefix", "pi_edit_normalize", "pi_json_reserialize")
PI_EDIT_TOOL = "edit"
# What JavaScript's String.prototype.trim removes (pi-ai's parseToolCallArguments trims).
_JS_BLANK = re.compile(
    r"[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]*"
)
_JSON_ESCAPES = '"\\/bfnrt'
_JSON_CONTROL_ESCAPES = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _pi_repair_json(text: str) -> str:
    """pi-ai's ``repairJson``, which runs on arguments JSON.parse cannot read.

    Inside string literals it escapes raw control characters and doubles a backslash that starts
    no valid escape.
    """
    out, in_string, index = [], False, 0
    while index < len(text):
        char = text[index]
        if not in_string:
            out.append(char)
            in_string = char == '"'
        elif char == '"':
            out.append(char)
            in_string = False
        elif char == "\\":
            following = text[index + 1] if index + 1 < len(text) else None
            digits = text[index + 2 : index + 6]
            if following == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", digits):
                out.append("\\u" + digits)
                index += 5
            elif following is not None and following in _JSON_ESCAPES:
                out.append("\\" + following)
                index += 1
            else:
                out.append("\\\\")
        elif ord(char) <= 0x1F:
            out.append(_JSON_CONTROL_ESCAPES.get(char, f"\\u{ord(char):04x}"))
        else:
            out.append(char)
        index += 1
    return "".join(out)


def pi_tool_arguments(text: str) -> dict:
    """The arguments object pi keeps for a call whose arguments it received as ``text``.

    pi-ai 1.4.0 ``parseToolCallArguments`` (dist/utils/json-parse.js), which every pi API uses
    once a call is complete: JavaScript's JSON.parse, where a duplicate key keeps its last value;
    failing that, the same after ``repairJson``; a blank text, or one that is still not a JSON
    object, is kept as ``{}`` (with an error the tool reports). pi sends this object back as
    ``JSON.stringify`` of it (openai-completions.js, openai-responses-shared.js) or as it is
    (anthropic-messages.js).
    """
    if not isinstance(text, str) or _JS_BLANK.fullmatch(text):
        return {}
    try:
        value = json.loads(text, parse_constant=_no_constant)
    except ValueError:
        repaired = _pi_repair_json(text)
        if repaired == text:
            return {}
        try:
            value = json.loads(repaired, parse_constant=_no_constant)
        except ValueError:
            return {}
    return _js_stringified(value) if isinstance(value, dict) else {}


def _js_stringified(value: Any) -> Any:
    """A parsed value as JSON.stringify writes it: a number too large for a double is null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _js_stringified(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_js_stringified(v) for v in value]
    return value


def _single_pi_edit(value: Any) -> bool:
    """One edit as pi's ``edit`` tool takes it (``isSingleEditInput``): oldText and newText."""
    return (isinstance(value, dict) and isinstance(value.get("oldText"), str)
            and isinstance(value.get("newText"), str))  # fmt: skip


def _no_constant(name: str) -> Any:
    raise ValueError("not JSON: " + name)  # JavaScript's JSON.parse has no NaN or Infinity


def pi_edit_stored(args: Any) -> Any:
    """The arguments of a call to pi's ``edit`` tool as pi keeps them in its history.

    pi 1.4.0 prepares them before running the call (pi-coding-agent dist/core/tools/edit.js,
    ``prepareEditArguments``), changing in place the object its history holds: ``edits`` given as a
    JSON string becomes what it parses to when that is an array or one edit (``[edit]``), and one
    edit object becomes ``[edit]``. Anything else is left as it was.
    """
    if not isinstance(args, dict):
        return args
    edits = args.get("edits")
    if isinstance(edits, str):
        try:
            parsed = json.loads(edits, parse_constant=_no_constant)
        except ValueError:
            return args
        if isinstance(parsed, list):
            return dict(args, edits=parsed)
        return dict(args, edits=[parsed]) if _single_pi_edit(parsed) else args
    return dict(args, edits=[edits]) if _single_pi_edit(edits) else args


def pi_edit_prepared(args: Any) -> Any:
    """The arguments pi runs a call to its ``edit`` tool with (``prepareEditArguments``).

    The stored form, and then a legacy top-level ``oldText``/``newText`` pair appended to
    ``edits`` as one edit. That fold returns a new object, so pi's history keeps the pair.
    """
    args = pi_edit_stored(args)
    if not (isinstance(args, dict) and isinstance(args.get("oldText"), str)
            and isinstance(args.get("newText"), str)):  # fmt: skip
        return args
    edits = list(args["edits"]) if isinstance(args.get("edits"), list) else []
    edits.append({"oldText": args["oldText"], "newText": args["newText"]})
    rest = {k: v for k, v in args.items() if k not in ("oldText", "newText")}
    return dict(rest, edits=edits)


def rewritten_digest(payload: Optional[str], schema: Optional[dict], rewrites,
                     name: Optional[str] = None) -> Optional[str]:  # fmt: skip
    """Digest of function arguments after the echo rewrites a harness profile declares.

    ``schema_defaults``: top-level arguments set to the ``default`` their schema
    declares are left out, present or not. ``cd_cwd_prefix``: a leading
    ``cd <dir> &&`` is cut from a string ``command`` argument. ``pi_edit_normalize``:
    the arguments of a call to pi's ``edit`` tool (``name``) as pi keeps them
    (``pi_edit_stored``). ``pi_json_reserialize``: the payload read as pi reads it
    (``pi_tool_arguments``), not as strict JSON. None when the payload is not a JSON object.
    """
    if payload is None:
        return None
    if "pi_json_reserialize" in rewrites:
        value = pi_tool_arguments(payload)
    else:
        try:
            value = strict_json(payload) if payload.strip() else {}
        except ValueError:
            return None
    if not isinstance(value, dict):
        return None
    if "pi_edit_normalize" in rewrites and name == PI_EDIT_TOOL:
        value = pi_edit_stored(value)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if "schema_defaults" in rewrites and isinstance(properties, dict):
        value = {
            key: item
            for key, item in value.items()
            if not (
                isinstance(properties.get(key), dict)
                and "default" in properties[key]
                and _canonical(properties[key]["default"]) == _canonical(item)
            )
        }
    command = value.get("command")
    if "cd_cwd_prefix" in rewrites and isinstance(command, str):
        match = CD_PREFIX.match(command)
        if match and match.end() < len(command):
            value = dict(value, command=command[match.end() :])
    return "json:" + sha256_text(json.dumps(_canonical(value), separators=(",", ":")))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


TRANSPORT_KEYS = ("cache_control",)


def _semantic(value: Any) -> Any:
    """A message without transport hints; a string content equals one text part."""
    if isinstance(value, dict):
        out = {k: _semantic(v) for k, v in value.items() if k not in TRANSPORT_KEYS}
        if isinstance(out.get("content"), str):
            out["content"] = [{"type": "text", "text": out["content"]}]
        return out
    if isinstance(value, list):
        return [_semantic(v) for v in value]
    return value


def openai_error(body: dict) -> Optional[dict]:
    """The error object of an OpenAI-style error body, or None when it has no usable one.

    OpenAI's envelope is ``{"error": {"message", "type", "param", "code"}}``; the
    vLLM-style top-level ``{"object": "error", "message", "type", "code"}`` is read by
    the OpenAI SDKs too. A bare string ``error`` carries no type or code to recover by.
    """
    error = body.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error
    if body.get("object") == "error" and isinstance(body.get("message"), str):
        return body
    return None


def message_digest(message: Any) -> str:
    """Digest of one history message, for continuity.

    Harnesses re-serialize earlier messages between turns: Claude Code sends a user
    message as text parts with ``cache_control`` once and as a plain string the next
    time. Neither changes what the message says, so neither changes the digest.
    """
    return sha256_text(
        json.dumps(_semantic(message), sort_keys=True, ensure_ascii=False, default=str)
    )


def text_blocks(items: List["Item"]) -> List[tuple]:
    """(kind, text digest, carrier) of the text and reasoning items, in order: the echo's unit.

    The carrier is compared by value (its digest) where the protocol gives one, as Anthropic's
    thinking signature; otherwise by whether it is present.
    """
    return [(i.kind, sha256_text(i.text) if i.text is not None else None,
             i.carrier_digest if i.carrier_digest is not None else i.carrier == "present")
            for i in items if i.kind in ("text", "reasoning")]  # fmt: skip


def tools_key(raw: Any) -> str:
    return hashlib.sha256(
        json.dumps(raw, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def tool_declarations(entries: Any) -> Tuple[Tuple[str, str, int], ...]:
    """Every tool a list of declarations declares, as ``(name, digest, size)`` (USG-08).

    A namespace's tools are named ``<namespace>.<name>``, and each carries the namespace's own
    fields: a template may render them once per tool (DeepSeek V4.1 repeats the namespace's
    description in each). A declaration without a name is named by its type (Responses'
    ``web_search``). ``digest`` ignores key order; ``size`` is the UTF-8 length of the JSON.
    """
    declared = []
    for entry in entries if isinstance(entries, list) else ():
        namespace = isinstance(entry, dict) and entry.get("type") == "namespace"
        nested = entry.get("tools") if namespace else None
        if isinstance(nested, list) and nested:
            wrapper = {k: v for k, v in entry.items() if k != "tools"}
            for tool in nested:
                declared.append((f"{entry.get('name')}.{_declared_name(tool)}",
                                 tools_key({"namespace": wrapper, "tool": tool}),
                                 _json_size(wrapper) + _json_size(tool)))  # fmt: skip
        else:
            declared.append((_declared_name(entry), tools_key(entry), _json_size(entry)))
    return tuple(declared)


def _declared_name(spec: Any) -> str:
    """A declaration's tool name: its own, its function's (Chat), else its type."""
    if isinstance(spec, dict):
        function = spec.get("function") if isinstance(spec.get("function"), dict) else {}
        for name in (spec.get("name"), function.get("name"), spec.get("type")):
            if isinstance(name, str) and name:
                return name
    return "?" + tools_key(spec)[:16]


def _json_size(value: Any) -> int:
    """UTF-8 length of a value's JSON."""
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8", "surrogatepass"))


def nested_from_descriptions(tools: Dict[str, Tool]) -> Dict[str, str]:
    """Tools that a custom tool declares inside its description (e.g. Codex's ``exec``)."""
    nested = {}
    for name, tool in tools.items():
        if tool.kind == "custom":
            for m in NESTED_HEADER.finditer(tool.description or ""):
                nested.setdefault(m.group(1), name)
    return nested


# --------------------------------------------------------------------------- calls

_GRAMMARS: Dict[str, Any] = {}


def payload_check(
    payload: Optional[str], tool: Tool, tolerate: Sequence[str] = ()
) -> Tuple[Optional[bool], Optional[dict]]:
    """(ok, detail) for one call payload against its declaration. detail never holds a value.

    ``tolerate`` names JSON-schema validators (``additionalProperties``) whose violations the
    harness tolerates: a payload whose every violation is one of them is ok, with detail
    ``{"reason": "tolerated", ...}``. It may also name a repair the harness makes before it runs
    a call: ``pi_edit_normalize``, pi's own preparation of its ``edit`` tool's arguments
    (``pi_edit_prepared``). A payload invalid as written is then ok when the repaired one is
    valid, with ``"repair"`` in the detail.
    """
    if tool.kind == "function":
        if not isinstance(payload, str):
            return False, {"reason": "arguments_not_string"}
        try:
            args = strict_json(payload) if payload.strip() else {}
        except ValueError as error:
            return False, {"reason": "arguments_not_strict_json", "error": type(error).__name__}
        if not isinstance(args, dict):
            return False, {"reason": "arguments_not_object"}
        if isinstance(tool.schema, dict):
            if jsonschema is None:
                return True, {"reason": "schema_unchecked_no_jsonschema"}
            try:
                validator = jsonschema.validators.validator_for(tool.schema)
                validator.check_schema(tool.schema)
                errors = list(validator(tool.schema).iter_errors(args))
            except SCHEMA_UNUSABLE:
                return True, {"reason": "declared_schema_invalid"}
            if errors:
                error = jsonschema.exceptions.best_match(errors)
                detail = {
                    "validator": error.validator,
                    "path": "/".join(str(p) for p in error.absolute_path),
                }
                if all(e.validator in tolerate for e in errors):
                    return True, dict(detail, reason="tolerated")
                if "pi_edit_normalize" in tolerate and tool.name == PI_EDIT_TOOL:
                    try:  # the repaired arguments can reach a $ref the original ones did not
                        repaired = validator(tool.schema).iter_errors(pi_edit_prepared(args))
                        repaired_valid = next(repaired, None) is None
                    except SCHEMA_UNUSABLE:
                        return True, {"reason": "declared_schema_invalid"}
                    if repaired_valid:
                        return True, dict(detail, reason="tolerated", repair="pi_edit_normalize")
                return False, dict(detail, reason="schema")
        return True, None
    if tool.kind == "custom":
        if not isinstance(payload, str):
            return False, {"reason": "input_not_string"}
        fmt = tool.grammar or {}
        if fmt.get("type") == "grammar" and fmt.get("syntax") == "lark":
            if lark is None:
                return bool(payload), {"reason": "grammar_unchecked_no_lark"}
            grammar = fmt.get("definition") or ""
            if grammar not in _GRAMMARS:
                try:
                    _GRAMMARS[grammar] = lark.Lark(grammar, parser="earley")
                except lark.exceptions.LarkError:
                    _GRAMMARS[grammar] = None
            parser = _GRAMMARS[grammar]
            if parser is None:
                return True, {"reason": "declared_grammar_invalid"}
            try:
                parser.parse(payload)
            except lark.exceptions.LarkError as error:
                return False, {"reason": "grammar", "error": type(error).__name__}
            return True, None
        return (bool(payload), None if payload else {"reason": "empty_input"})
    return None, {"reason": "unsupported_declared_type", "declared_type": tool.kind}


def classify_call(item: Item, request: RequestView, tolerate: Sequence[str] = ()) -> dict:
    """One class for one delivered call, against its own request's declarations.

    Classes: valid; serving shape (type_mismatch, namespace_mismatch,
    prefixed_name_of_declared); model (undeclared, nested_tool_called_directly,
    payload_invalid); payload_tolerated (invalid only in ways the harness tolerates,
    ``tolerate``: counted, not judged); unsupported_declared_type (counted, not judged);
    server_tool_called (a server-executed tool emitted as a client call).
    """
    name = item.name or ""
    tool = request.tools.get(name)
    result = {"class": "valid", "call_kind": item.call_kind}
    if tool is None:
        qualified = bool(QUALIFIED_NAME.match(name))
        tail = name.rsplit(".", 1)[1] if qualified else None
        if name in request.nested or (tail and tail in request.nested):
            result["class"] = "nested_tool_called_directly"
        elif tail and tail in request.tools:
            result["class"] = "prefixed_name_of_declared"
        else:
            result["class"] = "undeclared"
            if name and not qualified and not re.match(r"^[A-Za-z0-9_-]+$", name):
                result["detail"] = {"reason": "malformed_name"}
        return result
    if tool.kind == "server":
        result["class"] = "server_tool_called"
        return result
    if tool.kind == "builtin":
        result["class"] = "unsupported_declared_type"
        return result
    expected = {"function": "function", "custom": "custom"}.get(tool.kind)
    if expected != item.call_kind:
        result["class"] = "type_mismatch"
        result["declared_kind"] = tool.kind
        return result
    if item.namespace is not None and item.namespace != tool.namespace:
        result["class"] = "namespace_mismatch"
        return result
    ok, detail = payload_check(item.payload, tool, tolerate)
    if detail:
        result["detail"] = detail
    if ok is None:
        result["class"] = "unsupported_declared_type"
    elif ok and (detail or {}).get("reason") == "tolerated":
        result["class"] = "payload_tolerated"
    else:
        result["class"] = "valid" if ok else "payload_invalid"
    return result


def marker_hits(text: str, patterns) -> List[Tuple[str, bool]]:
    """(marker, quoted) per hit. Quoted means between quotes or backticks: usually prose."""
    hits = []
    for rx in patterns:
        for m in rx.finditer(text or ""):
            before = text[m.start() - 1] if m.start() > 0 else ""
            after = text[m.end()] if m.end() < len(text) else ""
            hits.append(
                (m.group(0), bool(before) and bool(after) and before in QUOTES and after in QUOTES)
            )
    return hits


def safe_name(name: Any) -> dict:
    """A tool name fit for a report: verbatim when it is an identifier, else only its shape."""
    if isinstance(name, str) and re.match(r"^[A-Za-z0-9_.-]{1,64}$", name):
        return {"name": name}
    text = name if isinstance(name, str) else ""
    return {"name_sha256_12": sha256_text(text)[:12], "name_chars": len(text)}


def content_text(value: Any) -> str:
    """Text of a string or a list of content parts ({"text": ...})."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            p.get("text", "")
            for p in value
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""
