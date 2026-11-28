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
"""Harness and model profiles, read the way the checks need them.

Profiles are data (``profiles/harness/*.yaml``, ``profiles/model/*.yaml``). A
field whose value starts with ``TODO`` is unknown: every helper here returns
``None`` for it, and the check that needs it reports BLOCKED, never PASS.
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Used when no model profile is given: markers of common tool-call and reasoning
# formats (XML-style tags, <|...|> special tokens, DeepSeek's DSML and tool tokens,
# [TOOL_CALLS]), so a missing profile does not silently disable the leak screen.
DEFAULT_MARKERS = (
    r"</?tool_call>",
    r"</?think>",
    r"</?arg_(?:key|value)>",
    r"<\|[a-z_]{2,24}\|>",
    r"<｜DSML｜",
    r"｜DSML｜",
    r"<｜tool▁",
    r"\[TOOL_CALLS\]",
)
DEFAULT_DEGENERATE = (r"(?:\[PAD\]){20,}",)


def is_todo(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper().startswith("TODO")


def known(value: Any) -> Any:
    """``value`` unless it is a TODO placeholder."""
    return None if is_todo(value) else value


class Profile(dict):
    """A loaded profile plus its path and SHA-256 (bound into every verdict)."""

    path: Optional[str] = None
    sha256: Optional[str] = None

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if is_todo(node) else node


def load_profile(path: Optional[str]) -> Profile:
    if not path:
        return Profile()
    with open(path, "rb") as fh:
        raw = fh.read()
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a profile must be a mapping")
    profile = Profile(data)
    profile.path = path
    profile.sha256 = hashlib.sha256(raw).hexdigest()
    return profile


# --------------------------------------------------------------------------- harness


# Where a harness writes what a tool result says about the call: at its start (an error's own
# message: pi's "Tool X not found", "Validation failed for tool", Node's "ENOENT: ...") or at its
# end (pi's bash status line, "Command exited with code N", appended after the command's output,
# which pi keeps up to 50 KB). Regexes and substrings are read in the first and the last
# OUTCOME_WINDOW characters: every marker a profile declares is one line, far shorter, so it is
# found whatever the result's length, and a long result is not scanned whole.
OUTCOME_WINDOW = 4096
# Joins the two windows of a long result: no anchor or `.` reaches across a line break.
_WINDOW_JOIN = "\n\n"


def outcome_view(text: str) -> str:
    """What outcome patterns read of a tool result: all of it, or its first and last windows."""
    if len(text) <= 2 * OUTCOME_WINDOW:
        return text
    return text[:OUTCOME_WINDOW] + _WINDOW_JOIN + text[-OUTCOME_WINDOW:]


class OutcomeRules:
    r"""Classifies a harness's tool result into an outcome class.

    Built from the harness profile's ``outcomes`` mapping, e.g.::

        aborted: {exact: aborted}
        unsupported: {prefix: "unsupported call:"}
        unavailable_in_mode: {regex: "is unavailable in (\\w+) mode"}
        error: {is_error: true}

    Rules are tried in profile order; the first match wins; no match is ``result``. A ``prefix``
    may be a list (any of them). ``is_error`` next to a text pattern is one more condition: the
    result must carry that flag (an absent flag is not ``true``). ``{none: reason}`` declares that
    the harness has no such outcome: it matches nothing, and a rated check of that class is
    NOT_APPLICABLE with the reason.
    """

    def __init__(self, rules: Dict[str, Dict[str, Any]]):
        self.rules = []
        for cls, rule in rules.items():
            if not isinstance(rule, dict) or "none" in rule:
                continue
            compiled = dict(rule)
            if "regex" in rule:
                compiled["_rx"] = re.compile(rule["regex"], re.I)
            if "prefix" in rule:
                prefix = rule["prefix"]
                compiled["prefix"] = tuple(prefix) if isinstance(prefix, list) else (prefix,)
            self.rules.append((cls, compiled))

    def classify(self, text: str, is_error: Optional[bool] = None) -> str:
        stripped = (text or "").strip()
        view = outcome_view(stripped)
        for cls, rule in self.rules:
            if "is_error" in rule and is_error is not rule["is_error"]:
                continue
            if "exact" in rule and stripped == rule["exact"]:
                return cls
            if "prefix" in rule and stripped.startswith(rule["prefix"]):
                return cls
            if "contains" in rule and rule["contains"] in view:
                return cls
            if "_rx" in rule and rule["_rx"].search(view):
                return cls
            if set(rule) == {"is_error"}:
                return cls
        return "result"


def outcome_rules(harness: Profile) -> Optional[OutcomeRules]:
    rules = known(harness.get("outcomes"))
    if not isinstance(rules, dict) or not rules:
        return None
    if any(is_todo(v) for v in rules.values()):
        return None
    return OutcomeRules(rules)


def _dig(obj: Any, dotted: str) -> Any:
    node = obj
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def extract_identity(harness: Profile, headers: Dict[str, str], body: Any, record: dict) -> dict:
    """session/thread/agent/... per the profile's ``identity`` spec, with trace fallbacks.

    Fallbacks, in order: the replay corpus thread, the request trace's ``thread``
    and ``session`` fields. A value that cannot be resolved stays None.
    """
    spec = harness.get("identity") if isinstance(harness.get("identity"), dict) else {}
    out = {}
    for field, how in spec.items():
        value = None
        if isinstance(how, dict) and not is_todo(how):
            if "header" in how:
                value = headers.get(str(how["header"]).lower())
            elif "body" in how and isinstance(body, dict):
                value = _dig(body, how["body"])
            elif "body_json" in how and isinstance(body, dict):
                raw = _dig(body, how["body_json"])
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        raw = None
                if isinstance(raw, dict):
                    value = raw.get(how.get("key"))
        out[field] = str(value) if value not in (None, "") else None
    replay = record.get("replay") if isinstance(record.get("replay"), dict) else {}
    trace_thread = record.get("thread") if isinstance(record.get("thread"), dict) else {}
    if not out.get("thread"):
        out["thread"] = (
            replay.get("thread")
            or trace_thread.get("thread_id")
            or out.get("session")
            or (str(record["session"]) if record.get("session") else None)
        )
    if not out.get("session"):
        out["session"] = str(record["session"]) if record.get("session") else out.get("thread")
    return out


# --------------------------------------------------------------------------- model


def client_cut(harness: Optional["Profile"], key: str):
    """A client timeout from the harness profile: ``(seconds, exact)``, or ``(None, None)``.

    A number is the cut itself. ``{at_least: s}`` is a lower bound: silences that long are
    known to survive, and the cut itself is unknown.
    """
    value = known(harness.get_path(key)) if harness else None
    if isinstance(value, bool):
        return None, None
    if isinstance(value, (int, float)):
        return float(value), True
    if isinstance(value, dict) and isinstance(value.get("at_least"), (int, float)) \
            and not isinstance(value.get("at_least"), bool):  # fmt: skip
        return float(value["at_least"]), False
    return None, None


def native_card(model: Optional["Profile"]) -> Optional[dict]:
    """The model's ``native`` section for the raw view, with its qualified-name prefix; None if unknown."""
    section = model.get("native") if model else None
    if not isinstance(section, dict) or is_todo(section.get("kind")):
        return None
    card = model.get("format_card") if isinstance(model.get("format_card"), dict) else {}
    return dict(section, qualified_prefix=card.get("qualified_prefix") or "functions.")


def prompt_offset(model: Optional["Profile"], protocol: Optional[str] = None) -> int:
    """Prompt tokens the model's usage accounting leaves out on purpose (model profile ``usage``).

    ``usage.prompt_offset`` is one number for every protocol, or a number per protocol
    (``responses``, ``anthropic-messages``, ``chat-completions``): a serving extension may adjust
    the usage of some routes only. A protocol it does not name has none.
    """
    section = model.get("usage") if model else None
    value = section.get("prompt_offset") if isinstance(section, dict) else None
    if isinstance(value, dict):
        value = value.get(protocol)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def marker_patterns(model: Profile) -> List["re.Pattern"]:
    markers = known(model.get("markers")) if model else None
    if not markers:
        return [re.compile(p) for p in DEFAULT_MARKERS]
    return [re.compile(re.escape(m)) for m in markers if isinstance(m, str) and m]


def degenerate_patterns(model: Profile) -> List["re.Pattern"]:
    entries = known(model.get("degenerate")) if model else None
    if entries is None or entries == []:
        return [re.compile(p) for p in DEFAULT_DEGENERATE]
    patterns = []
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("token"), str):
            run = int(entry.get("run") or 20)
            patterns.append(re.compile(f"(?:{re.escape(entry['token'])}){{{run},}}"))
        elif isinstance(entry, str):
            patterns.append(re.compile(entry))
    return patterns or [re.compile(p) for p in DEFAULT_DEGENERATE]


def degenerate_id_runs(model: Profile) -> List[Tuple[int, int]]:
    """(token id, run length) per degenerate entry that names its id: read on token ids.

    ``id: last_vocab_id`` resolves against the profile's ``vocab_size``; ``id_range: [a, b]``
    names ids a to b inclusive; an entry whose id cannot be resolved is left to the text
    patterns.
    """
    entries = known(model.get("degenerate")) if model else None
    vocab = model.get("vocab_size") if model else None
    runs = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        span = entry.get("id_range")
        if isinstance(span, list) and len(span) == 2 and all(isinstance(i, int) for i in span):
            # [first, last], inclusive: a block of ids, e.g. a vocabulary's untrained rows.
            runs += [(i, int(entry.get("run") or 20)) for i in range(span[0], span[1] + 1)]
            continue
        token_id = entry.get("id")
        if token_id == "last_vocab_id" and isinstance(vocab, int):
            token_id = vocab - 1
        if isinstance(token_id, int) and not isinstance(token_id, bool):
            runs.append((token_id, int(entry.get("run") or 20)))
    return runs


def degenerate_text_declared(model: Profile) -> bool:
    """Whether the profile names a degenerate token the delivered or raw text can show.

    Entries that name only an id are read on raw token ids, never on text.
    """
    entries = known(model.get("degenerate")) if model else None
    return any((isinstance(e, dict) and isinstance(e.get("token"), str)) or isinstance(e, str)
               for e in entries or [])  # fmt: skip


def has_id_run(token_ids: List[int], runs: List[Tuple[int, int]]) -> bool:
    """Whether ``token_ids`` holds ``run`` copies of ``id`` in a row, for any (id, run)."""
    needed = dict(runs)
    current, length = None, 0
    for token in token_ids:
        length = length + 1 if token == current else 1
        current = token
        if token in needed and length >= needed[token]:
            return True
    return False


# --------------------------------------------------------------------------- matrix


def load_thresholds(path: Optional[str]) -> Tuple[dict, Optional[str]]:
    """The ``thresholds`` section of a run's matrix file, or {} (every rate unknown).

    Also returns the SHA-256 of the bytes they were read from (None without a file).
    """
    if not path:
        return {}, None
    with open(path, "rb") as fh:
        raw = fh.read()
    data = yaml.safe_load(raw) or {}
    thresholds = data.get("thresholds") if isinstance(data, dict) else None
    return (thresholds if isinstance(thresholds, dict) else {}), hashlib.sha256(raw).hexdigest()
