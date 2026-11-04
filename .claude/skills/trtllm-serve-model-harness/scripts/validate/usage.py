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
"""Token usage per protocol (check C-14): present, typed, consistent, and true to other counts.

Harnesses steer by these numbers: Codex compacts on the reported total, Claude
Code sizes its context with ``input_tokens`` (and with ``count_tokens`` before a
turn), pi keeps a running total. A wrong count does not break the stream; it
makes the harness compact too early or overflow (references/protocols.md
section 6).

    USG-01  a field the protocol requires is missing, not an integer, or negative
    USG-02  the total is not the sum of its parts
    USG-03  a detail count exceeds the count it is part of
    USG-04  output was delivered, yet zero output tokens are reported
    USG-05  count_tokens disagrees with the input the same request was billed
    USG-06  the reported output differs from the engine's own count (raw record)
    USG-07  the reported input differs from the server's own prompt count (trace record)
    USG-08  the next turn billed less input though its prompt holds this turn's calls
"""

import json
from typing import List, Optional

from turnlib import ReplyView, Violation, _semantic, sha256_text

# The fields count_tokens counts: those of the Messages request that shape the rendered prompt
# (tensorrt_llm/serve/anthropic_adapter.py, convert_anthropic_count_tokens_request).
COUNTED_FIELDS = ("model", "messages", "system", "tools", "tool_choice", "thinking", "output_config",
                  "context_management")  # fmt: skip


def counted_digest(body: dict) -> str:
    """One digest for a Messages request and a count_tokens request that render the same prompt."""
    counted = {k: body.get(k) for k in COUNTED_FIELDS if body.get(k) is not None}
    return sha256_text(
        json.dumps(_semantic(counted), sort_keys=True, ensure_ascii=False, default=str)
    )


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def input_tokens(protocol: str, usage: dict) -> Optional[int]:
    """The whole prompt as the protocol reports it (Anthropic splits off cache reads and writes)."""
    if not isinstance(usage, dict):
        return None
    if protocol == "anthropic-messages":
        parts = [usage.get(k) for k in ("input_tokens", "cache_read_input_tokens",
                                        "cache_creation_input_tokens")]  # fmt: skip
        return sum(p for p in parts if _int(p)) if _int(parts[0]) else None
    value = usage.get("prompt_tokens" if protocol == "chat-completions" else "input_tokens")
    return value if _int(value) else None


def output_tokens(protocol: str, usage: dict) -> Optional[int]:
    if not isinstance(usage, dict):
        return None
    value = usage.get("completion_tokens" if protocol == "chat-completions" else "output_tokens")
    return value if _int(value) else None


def _produced(items) -> bool:
    return any((i.kind == "call") or (i.text or "").strip() for i in items)


def rules(
    protocol: str, reply: ReplyView, items: list, include_usage: Optional[bool]
) -> List[Violation]:
    """USG-01 .. USG-04 for one reply."""
    if reply.kind not in ("sse", "json") or (
        isinstance(reply.http_status, int) and reply.http_status >= 400
    ):
        return []
    terminal = reply.terminal.get("kind")
    usage = reply.usage if isinstance(reply.usage, dict) else {}
    out: List[Violation] = []
    if protocol == "chat-completions":
        if not usage:
            if include_usage and terminal in ("completed", "incomplete") and reply.kind == "sse":
                out.append(Violation("USG-01", {"why": "requested_usage_missing"}))
            if reply.kind == "json" and terminal in ("completed", "incomplete"):
                out.append(Violation("USG-01", {"why": "usage_missing"}))
            return out
        names = ("prompt_tokens", "completion_tokens", "total_tokens")
        details = (("prompt_tokens_details", "cached_tokens", "prompt_tokens"),
                   ("completion_tokens_details", "reasoning_tokens", "completion_tokens"))  # fmt: skip
    elif protocol == "responses":
        if terminal not in ("completed", "incomplete"):
            return []
        names = ("input_tokens", "output_tokens", "total_tokens")
        details = (("input_tokens_details", "cached_tokens", "input_tokens"),
                   ("output_tokens_details", "reasoning_tokens", "output_tokens"))  # fmt: skip
    else:
        if terminal not in ("completed", "incomplete"):
            return []
        names = ("input_tokens", "output_tokens")
        details = ()
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if usage.get(key) is not None and not _int(usage[key]):
                out.append(Violation("USG-01", {"why": "type:" + key}))
    for name in names:
        if name not in usage or usage[name] is None:
            out.append(Violation("USG-01", {"why": "missing:" + name}))
        elif not _int(usage[name]):
            out.append(Violation("USG-01", {"why": "type:" + name}))
    if out:
        return out
    if len(names) == 3 and usage[names[2]] != usage[names[0]] + usage[names[1]]:
        out.append(Violation("USG-02", {"why": "total"}))
    for group, key, whole in details:
        part = (usage.get(group) or {}).get(key) if isinstance(usage.get(group), dict) else None
        if part is not None and (not _int(part) or part > usage[whole]):
            out.append(Violation("USG-03", {"why": key}))
    if _produced(items) and output_tokens(protocol, usage) == 0:
        out.append(Violation("USG-04", {"why": "zero_output"}))
    return out
