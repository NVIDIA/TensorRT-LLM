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
"""The model's own text, read in the model's own format (the raw view, check C-13).

A raw record (``raw-*.jsonl``) holds what the model generated, detokenized,
before any parser of trtllm-serve touched it. This module reads that text into
reasoning, content and calls, following the format each checkpoint's encoder
renders (the grammar is the encoder's, never trtllm-serve's parser's), so that
what the parsers delivered can be held against what the model wrote.

Readers never raise on malformed text. They return what they could read plus
``anomalies``: the model wrote something outside its own format, which makes
the turn the model's to answer for (validate.md section 2), not serving's. A
slip that serving repairs, and that the model profile declares in
``native.repairs``, is read the way serving reads it and named in ``repairs``
instead.

Kinds (model profile ``native.kind``):

  kimi_xtml  Kimi-K3, encoding_k3.py: <|open|>TAG ATTRS<|sep|> ... <|close|>TAG<|sep|>
  glm_xml    GLM-5.x, chat_template.jinja: </think>, <tool_call>NAME<arg_key>..</tool_call>
  dsml       DeepSeek V4.1, encoding/encoding.py: </think>, <｜DSML｜ calls> blocks
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from turnlib import Violation, _canonical

# A tool name as a model may write it: a declared name, possibly namespace-qualified.
CALL_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.:-]*$")
JSON_TYPES = {"number": (int, float), "boolean": (bool,), "null": (type(None),), "object": (dict,),
              "array": (list,)}  # fmt: skip


def _check_value(text: str, kind: Optional[str], anomalies: List[str]) -> None:
    """A value the format declares as JSON (of a given type) that is not: the model's slip."""
    if kind in (None, "string"):
        return
    try:
        value = json.loads(text)
    except ValueError:
        anomalies.append("argument_not_its_type")
        return
    expected = JSON_TYPES.get(kind)
    if expected and (
        not isinstance(value, expected) or (kind == "number" and isinstance(value, bool))
    ):
        anomalies.append("argument_not_its_type")


@dataclass
class NativeCall:
    name: str
    # (key, value text as written, declared type or None). The type is "string" when the
    # format says the value is a raw string, a JSON type name when it says the value is JSON,
    # and None when the format does not say (GLM writes strings bare and other values as JSON).
    args: List[Tuple[str, str, Optional[str]]] = field(default_factory=list)
    json_block: Optional[str] = None  # K3's <json> form: the arguments as one JSON text


@dataclass
class NativeTurn:
    reasoning: Optional[str]  # None: no reasoning channel in this text
    content: str
    calls: List[NativeCall]
    anomalies: List[str] = field(default_factory=list)
    repairs: List[str] = field(default_factory=list)  # declared repairs this reading applied


def read(native: dict, text: str) -> NativeTurn:
    """Read one generation with the model profile's ``native`` section."""
    kind = (native or {}).get("kind")
    reader = {"kimi_xtml": _kimi_xtml, "glm_xml": _glm_xml, "dsml": _dsml}.get(kind)
    if reader is None:
        raise ValueError(f"no native reader for kind {kind!r}")
    return reader(native, text or "")


def _strip_stops(native: dict, text: str, anomalies: List[str]) -> str:
    """Drop the stop markers the engine may leave at the end; anywhere else they are anomalies."""
    stops = [s for s in native.get("stop_markers") or [] if s]
    changed = True
    while changed:
        changed = False
        for stop in stops:
            if text.endswith(stop):
                text = text[: -len(stop)]
                changed = True
    if any(stop in text for stop in stops):
        anomalies.append("stop_marker_inside")
    return text


# --------------------------------------------------------------------------- Kimi-K3 XTML

_XTML = re.compile(r"<\|open\|>(.*?)<\|sep\|>|<\|close\|>(.*?)<\|sep\|>|<\|end_of_msg\|>", re.S)
_ATTR = re.compile(r'\s+([A-Za-z_][\w-]*)="([^"]*)"')


def _xtml_head(head: str) -> Tuple[str, dict]:
    name = head.split(None, 1)[0] if head.strip() else ""
    rest = head[len(name) :]
    attrs = {k: v.replace("&quot;", '"').replace("&amp;", "&") for k, v in _ATTR.findall(rest)}
    return name, attrs


def _kimi_xtml(native: dict, text: str) -> NativeTurn:
    anomalies: List[str] = []
    text = _strip_stops(native, text, anomalies)
    reasoning_parts: List[str] = []
    content_parts: List[str] = []
    calls: List[NativeCall] = []
    saw_think = False
    # The generation prompt opens the message and, in thinking mode, the think channel; the
    # first close marker shows which one the text starts in.
    first_close = re.search(r"<\|close\|>(\w+)<\|sep\|>", text)
    first_open = re.search(r"<\|open\|>", text)
    stack: List[Tuple[str, dict]] = [("message", {})]
    if first_close and (not first_open or first_close.start() < first_open.start()):
        if first_close.group(1) in ("think", "response"):
            stack.append((first_close.group(1), {}))
    elif native.get("prompt_opens") in ("think", "response"):
        stack.append((native["prompt_opens"], {}))
    saw_think = any(tag == "think" for tag, _ in stack)
    pos = 0
    current_call: Optional[NativeCall] = None
    current_arg: Optional[dict] = None

    def take_text(chunk: str) -> None:
        nonlocal current_arg
        if not chunk:
            return
        tag = stack[-1][0] if stack else None
        if tag == "think":
            reasoning_parts.append(chunk)
        elif tag == "response":
            content_parts.append(chunk)
        elif tag == "argument" and current_arg is not None:
            current_arg["value"] += chunk
        elif tag == "json" and current_call is not None:
            current_call.json_block = (current_call.json_block or "") + chunk
        elif chunk.strip():
            anomalies.append(f"stray_text_in_{tag or 'top'}")

    for match in _XTML.finditer(text):
        take_text(text[pos : match.start()])
        pos = match.end()
        if match.group(0) == "<|end_of_msg|>":
            if pos != len(text):
                anomalies.append("text_after_end_of_msg")
            break
        if match.group(1) is not None:  # open
            tag, attrs = _xtml_head(match.group(1))
            parent = stack[-1][0] if stack else None
            if tag == "think":
                saw_think = True
            if tag == "call":
                if parent != "tools":
                    anomalies.append("call_outside_tools")
                current_call = NativeCall(name=attrs.get("tool", ""))
                if not CALL_NAME.match(current_call.name):
                    anomalies.append("call_name_shape")
                if attrs.get("index") != str(len(calls) + 1):
                    anomalies.append("call_index")
                calls.append(current_call)
            elif tag == "argument":
                if parent != "call" or current_call is None:
                    anomalies.append("argument_outside_call")
                current_arg = {"key": attrs.get("key", ""), "type": attrs.get("type"), "value": ""}
            elif tag == "json":
                if parent != "call" or current_call is None:
                    anomalies.append("json_outside_call")
            elif tag not in ("think", "response", "tools", "message"):
                anomalies.append("unknown_tag")
            stack.append((tag, attrs))
        else:  # close
            tag = match.group(2).strip()
            if not stack or stack[-1][0] != tag:
                anomalies.append("close_mismatch")
                while stack and stack[-1][0] != tag:
                    stack.pop()
                if not stack:
                    stack = [("message", {})]
                    continue
            stack.pop()
            if tag == "argument" and current_arg is not None and current_call is not None:
                current_call.args.append(
                    (current_arg["key"], current_arg["value"], current_arg["type"])
                )
                _check_value(current_arg["value"], current_arg["type"], anomalies)
                current_arg = None
            elif tag == "json" and current_call is not None:
                _check_value(current_call.json_block or "", "object", anomalies)
            elif tag == "call":
                current_call = None
            if not stack:
                stack = [("message", {})]
    else:
        take_text(text[pos:])
    # The engine stops inside `tools` (after a call's close, the single-call stop) or after the
    # message's close; only a channel that carries text being left open is the model's doing.
    open_tags = [tag for tag, _ in stack[1:] if tag not in ("tools", "message")]
    if open_tags:
        anomalies.append("unclosed_" + open_tags[-1])
    return NativeTurn(
        reasoning="".join(reasoning_parts) if saw_think else None,
        content="".join(content_parts),
        calls=calls,
        anomalies=anomalies,
    )


# --------------------------------------------------------------------------- GLM


def _glm_owed_close(text: str, start: int, think_close: str, call_open: str,
                    call_close: str) -> int:  # fmt: skip
    """The first think close at or after ``start`` that is outside a call, or -1.

    A call runs from its opener to its close, or to the end when it is never closed; a think
    close inside it is part of the call's text.
    """
    pos = start
    while True:
        close = text.find(think_close, pos)
        opened = text.find(call_open, pos)
        if close < 0:
            return -1
        if opened < 0 or close < opened:
            return close
        end = text.find(call_close, opened + len(call_open))
        if end < 0:
            return -1
        pos = end + len(call_close)


def _glm_xml(native: dict, text: str) -> NativeTurn:
    anomalies: List[str] = []
    repairs: List[str] = []
    text = _strip_stops(native, text, anomalies)
    think_open, think_close = (
        native.get("think_open", "<think>"),
        native.get("think_close", "</think>"),
    )
    call_open, call_close = (
        native.get("call_open", "<tool_call>"),
        native.get("call_close", "</tool_call>"),
    )
    reasoning: Optional[str] = None
    rest = text
    starts_in_think = native.get("prompt_opens") == "think" or think_close in text
    if starts_in_think:
        cut = text.find(think_close)
        opener = text.find(call_open)
        owed = 0 <= opener < cut
        if owed:
            # A call opened inside the reasoning ends it there. The reasoning still owes its
            # close: the first one outside a call, which the model writes after the calls.
            cut = _glm_owed_close(text, opener, think_close, call_open, call_close)
        if cut < 0:
            return NativeTurn(
                reasoning=text, content="", calls=[], anomalies=anomalies + ["unclosed_think"]
            )
        if owed:
            reasoning, rest = text[:opener], text[opener:cut] + text[cut + len(think_close) :]
            # Serving that ends the reasoning at the call and drops the owed close delivers this
            # reading; a profile that does not declare it leaves the slip to the model.
            if "owed_think_close" in (native.get("repairs") or []):
                repairs.append("owed_think_close")
            else:
                anomalies.append("call_in_reasoning")
        else:
            reasoning, rest = text[:cut], text[cut + len(think_close) :]
        if reasoning.startswith(think_open):
            reasoning = reasoning[len(think_open) :]
            anomalies.append("think_reopened")
    if think_close in rest or think_open in rest:
        anomalies.append("think_marker_after_reasoning")
    pieces = rest.split(call_open)
    content = pieces[0]
    calls: List[NativeCall] = []
    for index, piece in enumerate(pieces[1:]):
        if call_close in piece:
            inner, after = piece.split(call_close, 1)
        else:
            inner, after = piece, ""
            anomalies.append("unclosed_call")
        if after.strip():
            anomalies.append("text_after_call")
        calls.append(_glm_call(native, inner, anomalies))
    return NativeTurn(reasoning=reasoning, content=content, calls=calls, anomalies=anomalies,
                      repairs=repairs)  # fmt: skip


def _glm_call(native: dict, inner: str, anomalies: List[str]) -> NativeCall:
    key_open, key_close = native.get("key_open", "<arg_key>"), native.get("key_close", "</arg_key>")
    value_open, value_close = (
        native.get("value_open", "<arg_value>"),
        native.get("value_close", "</arg_value>"),
    )
    name, _, body = inner.partition(key_open)
    call = NativeCall(name=name.strip())
    if not CALL_NAME.match(call.name):
        anomalies.append("call_name_shape")
    if not _:
        return call
    for part in (key_open + body).split(key_open)[1:]:
        key, sep, after = part.partition(key_close)
        if not sep:
            anomalies.append("unclosed_arg_key")
            continue
        after = after.lstrip("\n")
        if not after.startswith(value_open):
            anomalies.append("arg_without_value")
            continue
        value, sep, tail = after[len(value_open) :].partition(value_close)
        if not sep:
            anomalies.append("unclosed_arg_value")
        if tail.strip():
            anomalies.append("text_between_args")
        if key_close in value or value_open in value:
            # The format's own argument markup inside a value (one closed with </arg_key> and
            # opened again): where the value ends is no longer the format's to say. The same
            # slip with a second </arg_value> is text_between_args.
            anomalies.append("markup_in_arg_value")
        call.args.append((key.strip(), value, None))
    return call


# --------------------------------------------------------------------------- DeepSeek DSML


def _dsml(native: dict, text: str) -> NativeTurn:
    anomalies: List[str] = []
    text = _strip_stops(native, text, anomalies)
    token = native.get("dsml_token", "｜DSML｜")
    think_close = native.get("think_close", "</think>")
    block_start = f"\n\n<{token} calls>"
    block_end = f"</{token} calls>"
    reasoning: Optional[str] = None
    rest = text
    if native.get("prompt_opens") == "think" or think_close in text:
        cut = text.find(think_close)
        if cut < 0:
            return NativeTurn(
                reasoning=text, content="", calls=[], anomalies=anomalies + ["unclosed_think"]
            )
        reasoning, rest = text[:cut], text[cut + len(think_close) :]
    cut = rest.find(block_start)
    if cut < 0:
        if f"<{token}" in rest:
            anomalies.append("calls_block_not_after_blank_line")
        return NativeTurn(reasoning=reasoning, content=rest, calls=[], anomalies=anomalies)
    content, block = rest[:cut], rest[cut + len(block_start) :]
    if block_end in block:
        block, tail = block.split(block_end, 1)
        if tail.strip():
            anomalies.append("text_after_calls")
    else:
        anomalies.append("unclosed_calls")
    calls: List[NativeCall] = []
    invoke = re.compile(
        rf'<{re.escape(token)} invoke name="(.*?)">\n(.*?)(</{re.escape(token)} invoke>|$)', re.S
    )
    param = re.compile(
        rf'<{re.escape(token)} parameter name="(.*?)" string="(true|false)">(.*?)</{re.escape(token)} parameter>',
        re.S,
    )
    consumed = 0
    for match in invoke.finditer(block):
        if block[consumed : match.start()].strip():
            anomalies.append("text_between_calls")
        consumed = match.end()
        if not match.group(3):
            anomalies.append("unclosed_invoke")
        call = NativeCall(name=match.group(1))
        if not CALL_NAME.match(call.name):
            anomalies.append("call_name_shape")
        body_end = 0
        for p in param.finditer(match.group(2)):
            if match.group(2)[body_end : p.start()].strip():
                anomalies.append("text_between_parameters")
            body_end = p.end()
            kind = "string" if p.group(2) == "true" else "json"
            call.args.append((p.group(1), p.group(3), kind))
            if kind == "json":
                _check_value(p.group(3), "json", anomalies)
        if match.group(2)[body_end:].strip():
            anomalies.append("text_between_parameters")
        calls.append(call)
    if block[consumed:].strip():
        anomalies.append("text_between_calls")
    return NativeTurn(reasoning=reasoning, content=content, calls=calls, anomalies=anomalies)


# --------------------------------------------------------------------------- raw view vs delivered


def _reading(turn: NativeTurn) -> dict:
    return {
        "reasoning": turn.reasoning,
        "content": turn.content,
        "calls": [(c.name, list(c.args), c.json_block) for c in turn.calls],
        "anomalies": sorted(set(turn.anomalies)),
        "repairs": sorted(set(turn.repairs)),
    }


def summary(native: dict, record: dict) -> dict:
    """What the pipeline keeps of one raw record: the reading, never the record.

    A model whose prompt opens in a different place when the request turns thinking off
    (``prompt_opens_without_thinking``) gets a second reading under that opening; the pipeline
    picks one per request.
    """
    text = record.get("text") or ""
    reading = _reading(read(native, text))
    other = native.get("prompt_opens_without_thinking")
    if other and other != native.get("prompt_opens"):
        reading["without_thinking"] = _reading(read(dict(native, prompt_opens=other), text))
    return reading


def _name_matches(native: dict, written: str, item) -> Optional[str]:
    """'' when the delivered call names what the model wrote; a repair's name; None otherwise."""
    prefix = native.get("qualified_prefix") or "functions."
    if written in (item.name, f"{item.namespace}.{item.name}" if item.namespace else None,
                   prefix + str(item.name)):  # fmt: skip
        return ""
    if "qualified_name_suffix" in (native.get("repairs") or []) and "." in written:
        if written.rsplit(".", 1)[1] == item.name:
            return "qualified_name_suffix"
    return None


def _args_equal(args: list, json_block: Optional[str], payload: Optional[str]) -> bool:
    """A delivered function payload against the arguments the model wrote, typed as written."""
    try:
        value = json.loads(payload or "{}")
    except ValueError:
        return False
    if not isinstance(value, dict):
        return False
    if json_block is not None:
        try:
            return _canonical(json.loads(json_block)) == _canonical(value)
        except ValueError:
            return False
    written = {key: (text, kind) for key, text, kind in args}
    if set(written) != set(value):
        return False
    for key, delivered in value.items():
        text, kind = written[key]
        if kind == "string" or (kind is None and isinstance(delivered, str)):
            if delivered != text:
                return False
            continue
        try:
            if _canonical(json.loads(text)) != _canonical(delivered):
                return False
        except ValueError:
            return False
    return True


def compare(native: dict, raw: dict, items: list, finished: bool, single_call: bool):
    """The raw reading against the delivered items of one reply.

    Returns ``(violations, notes)``. A violation means serving delivered something
    other than what the model wrote (rules RAW-01 .. RAW-05). Notes say why a
    difference is not serving's: ``anomaly:<kind>`` (the model broke its own
    format; nothing is judged), ``policy:first_call_only`` (parallel_tool_calls=false)
    or ``repair:<name>`` (a rewrite the model profile declares, of a call's name or
    of the reading itself).
    """
    if raw["anomalies"]:
        return [], ["anomaly:" + a for a in raw["anomalies"]]
    violations, notes = [], ["repair:" + r for r in raw.get("repairs") or ()]
    text = "".join(i.text or "" for i in items if i.kind == "text")
    reasoning = "".join(i.text or "" for i in items if i.kind == "reasoning" and i.text is not None)
    calls = [i for i in items if i.kind == "call"]
    written = list(raw["calls"])
    if single_call and len(written) > 1 and len(calls) == 1:
        written = written[:1]
        notes.append("policy:first_call_only")
    if not finished:
        # A reply the server stopped early carries a prefix of what the model wrote.
        prefix = (raw["content"].startswith(text) and (raw["reasoning"] or "").startswith(reasoning)
                  and len(calls) <= len(written)
                  and all(_name_matches(native, w[0], c) is not None for w, c in zip(written, calls)))  # fmt: skip
        if not prefix:
            violations.append(Violation("RAW-05", {"why": "stopped_reply_not_a_prefix"}))
        return violations, notes
    if len(calls) != len(written):
        violations.append(Violation("RAW-01", {"why": "call_count", "written": len(written),
                                               "delivered": len(calls)}))  # fmt: skip
    else:
        for (name, args, json_block), item in zip(written, calls):
            how = _name_matches(native, name, item)
            if how is None:
                violations.append(Violation("RAW-01", {"why": "call_name"}))
                continue
            if how:
                notes.append("repair:" + how)
            if item.call_kind == "custom":
                strings = [t for _, t, k in args if k in ("string", None)]
                if item.payload not in strings:
                    violations.append(Violation("RAW-02", {"why": "custom_input"}))
                elif len(args) > 1:
                    notes.append("custom:extra_written_arguments")
            elif not _args_equal(args, json_block, item.payload):
                violations.append(Violation("RAW-02", {"why": "arguments"}))
    if text != raw["content"]:
        violations.append(Violation("RAW-03", {"why": "whitespace" if text.strip() == raw["content"].strip()
                                               else "content"}))  # fmt: skip
    if reasoning != (raw["reasoning"] or ""):
        violations.append(Violation("RAW-04", {"why": "reasoning"}))
    return violations, notes
