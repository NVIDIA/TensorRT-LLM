<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harness contract: <harness name> <version>

Fill every section. Write "unknown" rather than leaving a section out; an unknown is a test to
write. Mark each line [measured], [source] or [doc].

## 1. Identity

- Harness, version, how it was installed, and the config used to point it at a self-hosted
  endpoint (file path and keys).
- Capture: command line, date, task(s) run, and capture method (`record_requests.py` record-only,
  forward mode, or trtllm-serve's request trace).

## 2. Wire

- API: Responses / Chat Completions / Anthropic Messages / other. Path(s) and method(s).
- Streaming: always, never, or configurable. What it does with a non-streamed reply.
- Body keys always present, sometimes present, and their enumerated values (from
  `summarize_capture.py`).
- Fields the server must accept even when it ignores them (e.g. `service_tier`,
  `client_metadata`, `store`, `include`).
- Auth: header name, and whether a key is required at all.

## 3. Tools

- How tools are declared: top-level `tools`, namespace groups, input items, system prompt text.
- Tool types: function (JSON schema), custom/freeform (grammar?), built-ins (web_search, shell,
  ...).
- Names: bare vs qualified spellings, and whether the harness accepts both.
- What the harness sends back for a call it cannot use (exact strings), and how a failed tool
  execution looks.
- `parallel_tool_calls` / `tool_choice` values and expected enforcement.

## 4. Conversation state

- What each request carries: full history every turn, `previous_response_id`, or a server store.
- How reasoning is replayed: visible text, summary, or an opaque carrier (`encrypted_content`).
  What the server must do with a carrier it did not write.
- Message roles and content part types (developer, system, images, files).
- Thread/session identity fields, and which one groups a conversation versus routes the cache.

## 5. Stream semantics

- Events the harness consumes and the one it records items from (e.g. `output_item.done`).
- What it requires of the terminal event; what it does with `incomplete` and `failed`.
- Idle timeouts: before the first event, and between events.
- Retries: which statuses and errors are retried, how many times, with what body.

## 6. Context window

- How the harness learns the window (config value, error code, usage numbers).
- What triggers compaction, and what the server must return for an over-length input.

## 7. Known quirks and checks to enable

- Model-agnostic quirks (e.g. sub-agents sharing a session key).
- Conformance checks to run (`references/validate.md`), with the markers and
  degenerate patterns for the model under test.
- Harness-side signals to count in smoke runs (`references/smoke.md`): aborted calls,
  unsupported calls, resends, disconnects.

## 8. The harness profile derived from this contract

Write `profiles/harness/<id>.yaml` from the sections above, one profile per protocol choice and
configuration. The validator reads only the profile. Every field cites a contract section, and
any field still unknown is `TODO`. A `TODO` field makes the checks that need it BLOCKED, never
PASS.

| Profile field | From section |
|---|---|
| `protocol`, `routes` | 2 Wire |
| `identity` | 4 Conversation state |
| `request.must_accept`, `request.model_field` | 2 Wire |
| `tools`, `parallel_tool_calls`, `outcomes` | 3 Tools |
| `reasoning` | 4 Conversation state |
| `stream`, `timeouts` | 5 Stream semantics |
| `errors`, `context_window`, `compaction` | 5 and 6 |
