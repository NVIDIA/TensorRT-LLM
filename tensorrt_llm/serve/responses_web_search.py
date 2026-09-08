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
"""The Responses API's half of the ``web_search`` server tool.

Everything about actually searching - providers, budget, domain filtering -
lives in tensorrt_llm/serve/web_search.py, which is endpoint-neutral. This
module only covers what is specific to this API:
recognising the tool in a ResponsesRequest, describing it to the model, and
reporting a completed search as a ``web_search_call`` output item.

A chat template can only describe functions, so the tool is offered to the
model as a one-argument function and the call is intercepted here rather than
returned to the client. The client never sees a function call named
``web_search``: it asked the server to do the searching, and a call it has no
implementation for comes back as "unsupported call".
"""

import json
from typing import Any, List, Optional, Sequence, Tuple

from openai.types.responses.response_function_web_search import (
    ActionSearch,
    ResponseFunctionWebSearch,
)

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.web_search import SearchOutcome, WebSearchToolSpec, load_web_search_config

# How the tool is named to the model. The client's tool has no function name
# of its own - it is identified by type - so one is chosen here.
#
# The name is deliberately one a client cannot produce. split_web_search_calls
# routes a model-emitted call to the server or back to the client by comparing
# this name, and a client is free to declare its own function tool called
# "web_search". Sharing the name would send that client call to the server
# search instead, and the client would then wait forever for a result it is
# never handed. Reserving a dunder-prefixed name makes the collision
# unreachable rather than merely unlikely.
WEB_SEARCH_FUNCTION_NAME = "__trtllm_web_search"

# What the tool is called when reporting it back to the client or naming it in
# an error. This is the client's own vocabulary and must not be the internal
# routing name above.
WEB_SEARCH_PUBLIC_NAME = "web_search"

# The Responses API spells the built-in tool with a version suffix on some
# releases ("web_search_preview", "web_search_2025_03_11"), so the type is
# matched by prefix rather than by an exact list that would need updating.
_WEB_SEARCH_TYPE_PREFIX = "web_search"

_QUERY_ARG = "query"


def is_web_search_tool(tool: Any) -> bool:
    tool_type = getattr(tool, "type", None)
    return bool(tool_type) and str(tool_type).startswith(_WEB_SEARCH_TYPE_PREFIX)


def web_search_tool_spec(tools: Optional[Sequence[Any]]) -> Optional[WebSearchToolSpec]:
    """Reduce a request's tools to what the shared module needs, or None."""
    for tool in tools or []:
        if not is_web_search_tool(tool):
            continue
        filters = getattr(tool, "filters", None)
        allowed = getattr(filters, "allowed_domains", None) if filters else None
        blocked = getattr(filters, "blocked_domains", None) if filters else None
        return WebSearchToolSpec(
            name=getattr(tool, "name", None) or WEB_SEARCH_PUBLIC_NAME,
            type=str(getattr(tool, "type", "")) or None,
            max_uses=getattr(tool, "max_uses", None),
            allowed_domains=tuple(allowed or ()),
            blocked_domains=tuple(blocked or ()),
        )
    return None


def web_search_rejection_reason(tools: Optional[Sequence[Any]]) -> Optional[str]:
    """Why this request's web_search tool cannot be honoured, or None.

    web_search is a server tool: the client hands over the definition and
    expects this server to run the query and fold the results into the same
    response. When that cannot happen, answering anyway is the one outcome the
    client cannot detect - it receives a normal answer and has no way to know
    the model never searched. Rejecting is recoverable (drop the tool and
    retry); a silently unsearched answer is not.
    """
    if web_search_tool_spec(tools) is None:
        return None
    if load_web_search_config().enabled:
        # A provider is configured, so the operator does expect live search,
        # but the per-request search loop is not wired into this endpoint yet -
        # the pieces live here and in web_search.py without a driver.
        return (
            "a provider is configured but the per-request search loop is "
            "not wired into this endpoint yet"
        )
    return "no web search provider is configured on this server"


def server_executes_web_search(tools: Optional[Sequence[Any]]) -> bool:
    """Whether this request's web_search tool can actually be run here.

    Used to decide whether to describe the tool to the model at all. Offering
    a tool the server cannot execute wastes the turn: the model calls it and
    the answer never arrives.
    """
    if web_search_tool_spec(tools) is None:
        return False
    return load_web_search_config().enabled


def web_search_function_definition() -> dict:
    """The function the model is shown in place of the built-in tool."""
    return {
        "name": WEB_SEARCH_FUNCTION_NAME,
        "description": (
            "Search the web for current information. Use this when "
            "the answer depends on recent events or on facts you "
            "are unsure of."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                _QUERY_ARG: {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": [_QUERY_ARG],
        },
    }


def query_from_call(arguments: Optional[str]) -> Optional[str]:
    """Pull the query out of a model-emitted call, tolerating sloppy shapes.

    A model that answers with a bare string instead of an object still gets
    its search run; refusing would spend the turn on a formatting detail.
    """
    if not arguments:
        return None
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments.strip() or None
    if isinstance(parsed, dict):
        value = parsed.get(_QUERY_ARG)
        if value is None and len(parsed) == 1:
            value = next(iter(parsed.values()))
        return str(value).strip() if value is not None else None
    if isinstance(parsed, str):
        return parsed.strip() or None
    return None


def search_call_item(outcome: SearchOutcome) -> ResponseFunctionWebSearch:
    """Report a completed search to the client, in this API's shape.

    The client is told the search happened even when it failed; a search that
    silently vanishes leaves the answer looking unsourced.
    """
    from tensorrt_llm.serve.responses_utils import _random_uuid

    return ResponseFunctionWebSearch(
        id=f"ws_{_random_uuid()}",
        action=ActionSearch(query=outcome.query, type="search"),
        status="completed" if outcome.ok else "failed",
        type="web_search_call",
    )


def history_items_for_search(call_id: str, query: str, outcome: SearchOutcome) -> List[dict]:
    """The input items that carry one finished search into the next turn.

    The search is replayed as an ordinary function call and result, because
    that is the shape the chat template already renders; the client-facing
    ``web_search_call`` item is a separate concern (see search_call_item).
    """
    return [
        {
            "type": "function_call",
            "call_id": call_id,
            "name": WEB_SEARCH_FUNCTION_NAME,
            "arguments": json.dumps({_QUERY_ARG: query}),
        },
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": outcome.as_model_text(),
        },
    ]


def split_web_search_calls(output_items: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    """Separate the model's web_search calls from its other tool calls.

    A client tool call ends the search loop: it has to be returned to the
    client to execute, and continuing would strand it.
    """
    server_calls, client_calls = [], []
    for item in output_items:
        item_type = getattr(item, "type", None) or (
            item.get("type") if isinstance(item, dict) else None
        )
        if item_type != "function_call":
            continue
        name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None)
        if name == WEB_SEARCH_FUNCTION_NAME:
            server_calls.append(item)
        else:
            client_calls.append(item)
    return server_calls, client_calls


def log_search(outcome: SearchOutcome) -> None:
    if outcome.ok:
        logger.info("web search %r -> %d results", outcome.query, len(outcome.results))
    else:
        logger.warning("web search %r failed: %s", outcome.query, outcome.error)
