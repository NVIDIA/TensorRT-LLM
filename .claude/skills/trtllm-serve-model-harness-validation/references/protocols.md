<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wire rules per protocol

Validator-owned. These rules back check C-01 (`validate.md`), and each rule id appears in a
turn record's `reply.grammar` when it is broken. They cover what trtllm-serve sends. What a
particular harness does with it lives in that harness's contract and profile.

Sources:
- the OpenAI Responses and Chat Completions streaming formats;
- the Anthropic Messages streaming format;
- the invariants trtllm-serve itself states. The Anthropic stream reframer docstring is in
  `tensorrt_llm/serve/anthropic_adapter.py`; the Responses rules come from what
  `check_stream.py` enforced on production streams.

When a source and this file disagree, raise it as a policy change. Do not patch a check to
match the server.

## 1. Responses (`POST /v1/responses`, `stream: true`)

Every SSE frame is `event: <name>` plus `data: <json>` with the same `type`.

| Rule | Requirement | `check_stream.py` name |
|---|---|---|
| RSP-01 | Opens with `response.created`, whose `response` has an `id`, a `model` and `output: []`: SDK clients start their snapshot from that object. `response.in_progress` may follow and may repeat as a keepalive | — |
| RSP-02 | Exactly one terminal event: `response.completed`, `response.incomplete` or `response.failed` | `no_terminal_event`, `multiple_terminal_events` |
| RSP-03 | Nothing after the terminal event | `events_after_terminal` |
| RSP-04 | `sequence_number` runs 0, 1, 2, ... with no gap, as OpenAI and trtllm-serve number a stream (a server that adds keepalives renumbers at egress) | `sequence_not_increasing`, `not_from_zero`, `gap` |
| RSP-05 | One response id and one model across the opening and terminal events | `response_id_changes`, `model_changes` |
| RSP-06 | Every `output_item.added` has its `output_item.done`, and the reverse | `unpaired_output_item` |
| RSP-07 | Every `content_part.added` has its `content_part.done`, and the reverse | `unpaired_content_part` |
| RSP-08 | Joined deltas equal the matching `*.done` payload: `output_text`, `reasoning_text`, `function_call_arguments`, `custom_tool_call_input` | `delta_done_mismatch` |
| RSP-09 | Each `*.done` payload equals the text, arguments or input of its `output_item.done` item | `done_item_mismatch` |
| RSP-10 | The `(type, id)` list of `output_item.done` equals the terminal snapshot's `output` | `item_list_differs` |
| RSP-11 | Per item id, the snapshot's text, name, call_id, arguments and input equal the streamed item, and so do a reasoning item's `encrypted_content` and an item's `status` where the snapshot carries one (a client that keeps the snapshot replays them) | `item_content_differs` (names the fields) |
| RSP-17 | A repeated `response.in_progress` (a keepalive) equals the stream's first one apart from `sequence_number`: it says nothing new | `keepalive_differs` |
| RSP-12 | A snapshot `message` has `status`. `function_call` and `reasoning` may omit it but never carry an explicit null. `custom_tool_call` has no status field | `item_status_missing` |
| RSP-13 | A custom tool is delivered as `custom_tool_call` with `input`; a function as `function_call` with JSON `arguments` | check C-06 `type_mismatch` |
| RSP-14 | A `response.failed` snapshot carries no generated output: count it, do not compare items | `failed_snapshot_without_streamed_items` (counted) |
| RSP-16 | A JSON body is a Response object or an error with an error object carrying a string `message` (`{"error": {...}}`, or the vLLM-style top-level `{"object": "error", "message", ...}` that the OpenAI SDKs also read). Anything else, a bare string `error` included, is broken: the harness gets no type or code to recover by | — |
| RSP-15 | `output_index` and `content_index` address what the stream announced. `output_item.added` carries the next index (0, 1, 2 ...); every later event of that item carries the same index; `content_part.added` carries the item's next part index; an `output_text` event names a part that was added. SDK clients keep the output as a list and place each event by these indices, not by item id | — |

**Over-length input.** Answer it in the shape the harness profile's `errors.overlength` states
(check C-10). The usual Responses shape is in-stream: `response.failed` with
`error.code == "context_length_exceeded"`. For example, Codex compacts only on that, and an
HTTP 400 makes it resend.

## 2. Anthropic Messages (`POST /v1/messages`, `stream: true`)

Framing is `event: <type>` plus `data: <json>` whose `type` equals the event name (ANT-15). The
sequence:

```text
message_start
  (content_block_start  content_block_delta*  content_block_stop)*
message_delta
message_stop
```

`ping` may appear anywhere, and `error` ends the stream.

| Rule | Requirement |
|---|---|
| ANT-01 | The first non-`ping` event is `message_start`, exactly once. `message.role == "assistant"`, `message.content == []`, `message.id` and `message.model` are set, and `usage.input_tokens` is present |
| ANT-02 | `content_block_start` for index `i` comes before any delta or stop for `i`. Indices start at 0 and increase by 1 |
| ANT-03 | Blocks are sequential: block `i+1` starts only after `content_block_stop` for `i`. trtllm-serve states this invariant; the reframer serializes interleaved upstream tool calls |
| ANT-04 | The delta type matches the block type: `text` takes `text_delta`; `tool_use` takes `input_json_delta`; `thinking` takes `thinking_delta`, then optionally one `signature_delta`; `redacted_thinking` takes none |
| ANT-05 | Every started block is stopped. There are no deltas after a block's stop |
| ANT-06 | A `tool_use` start carries an `id` that is unique within the message, a `name`, and `input: {}`. The joined `partial_json` is a JSON object, and an empty string means `{}`; `NaN` and `Infinity` are not JSON |
| ANT-07 | `signature_delta`, if present, comes after the last `thinking_delta` of its block and before its stop. Whether a signature is required is harness policy (`reasoning.carrier` in the profile) |
| ANT-08 | Exactly one `message_delta`, after every block has stopped. `delta.stop_reason` is one of `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, `pause_turn` or `refusal`. `usage.output_tokens` is present |
| ANT-09 | `stop_reason == "tool_use"` exactly when there is at least one `tool_use` block, except when `max_tokens` cut the reply |
| ANT-10 | Exactly one `message_stop`, as the last event. A body that ends after a block has started but before `message_delta` reads to a client as a dropped connection (e.g. Claude Code) |
| ANT-11 | `error` events carry `{"type": "error", "error": {"type": T, "message": M}}`, with `T` one of `invalid_request_error`, `authentication_error`, `permission_error`, `not_found_error`, `request_too_large`, `rate_limit_error`, `api_error` or `overloaded_error`. Nothing follows an error |
| ANT-12 | Under `disable_parallel_tool_use: true`, at most one `tool_use` block (check C-07) |
| ANT-13 | Non-streaming bodies: `type: "message"`, `role: "assistant"`, a content block list, a non-null `stop_reason`, and `usage` |
| ANT-14 | HTTP errors use the same envelope, with matching statuses: 400 `invalid_request_error`, 401, 403, 404, 413 `request_too_large`, 429 `rate_limit_error`, 500 `api_error`, 529 `overloaded_error`. `retry-after` is an integer number of seconds |
| ANT-15 | Every frame has an `event:` line, and it names the data's `type`. The Python and TypeScript SDKs dispatch on the `event:` line and drop a frame without one, so such a reply reads as empty to a client built on them (e.g. Claude Code) |

**Liveness.** `ping` is the keepalive. A client may abort a stream when no bytes arrive within
its idle watchdog (e.g. Claude Code). trtllm-serve starts the message at the first upstream
chunk, after queueing, prefill and the first token, because that chunk carries `input_tokens`. A
ping before `message_start` is the only way to cover that wait (check R-01). Pings after it cover
silences such as a tool call the tool parser holds until it is complete (check R-02). Clients skip
pings wherever they appear: the Python SDK and the TypeScript SDK bundled in Claude Code
(2.1.309) drop them before their message accumulator.

**Gaps on main when these rules were written (not rules; re-check before relying on one):**
- no `ping` is emitted (the keepalive change pings every 10 s of silence, from the start of the
  stream);
- `signature_delta` is defined but never emitted;
- errors are always `api_error`;
- over-length input is an HTTP 400 with OpenAI wording;
- `tool_choice` `any`/`tool` and `disable_parallel_tool_use` are rejected with a 400.

Whether each one breaks a harness is an S0 measurement, recorded in its contract.

## 3. Chat Completions (`POST /v1/chat/completions`, `stream: true`)

Each frame is `data: <json>`. The stream ends with `data: [DONE]`.

| Rule | Requirement |
|---|---|
| CHT-01 | Every frame before `[DONE]` is a `chat.completion.chunk` object or an error object |
| CHT-02 | `id`, `model` and `created` are constant across chunks |
| CHT-03 | For each choice, the first delta carries `role: "assistant"` |
| CHT-04 | `tool_calls` fragments are keyed by `index`. A call's first fragment carries `id`, `type: "function"` and `function.name`, and its index is the next one (0, 1, 2 in order of first appearance): SDK clients address the `tool_calls` list by it. Later fragments carry no `id` and no `function.name` (`type` may repeat), because openai-python concatenates every string it receives again. Ids are unique |
| CHT-05 | The joined `function.arguments` of each index is a JSON object: JSON's grammar, which has no `NaN` or `Infinity` (Python's json.loads takes them, JavaScript's JSON.parse does not; a repeated key is grammatical, and M-01 counts it) |
| CHT-06 | Exactly one non-null `finish_reason` per choice, in that choice's last chunk, from `stop`, `length`, `tool_calls` and `content_filter`. Nothing follows it for that choice. Every choice is held to CHT-03 to CHT-07 on its own; the turn record is choice 0's |
| CHT-07 | `finish_reason == "tool_calls"` exactly when the choice has at least one tool call, except `length` |
| CHT-08 | A usage chunk (`choices: []`) appears only when `stream_options.include_usage` asked for it, at most once, after the finishing chunks |
| CHT-09 | `[DONE]` appears exactly once, last |
| CHT-10 | Reasoning travels in `delta.reasoning_content`, or in `delta.reasoning` for GPT-OSS. Which one the harness reads is in its profile (`reasoning.field`). Reasoning text never appears in `content` (check C-09) |
| CHT-11 | Under `parallel_tool_calls: false`, at most one tool call index (check C-07) |
| CHT-12 | Non-streaming bodies: `choices[].message` with `role`, `content`, optional `reasoning_content` and `tool_calls`, plus `finish_reason` |
| CHT-13 | trtllm-serve's own choice fields, when present, have their types: `avg_decoded_tokens_per_iter` is a number and `speculative_decoding` an object. `stop_reason` has CHT-16. Any other non-standard choice or delta field is unchecked (WIRE-05) |
| CHT-14 | The choice indices are exactly 0 to n-1, with `n` from the request (default 1) |
| CHT-15 | An error body carries an error object with a string `message`: `{"error": {"message", "type", "param", "code"}}`, or the vLLM-style top-level `{"object": "error", "message", ...}`. A bare string `error` is broken. main answered a request with an unknown field this way, with a server source path and line in the string |
| CHT-16 | `stop_reason`, trtllm-serve's and vLLM's report of the stop token id or the stop string that ended a choice, is null, an integer or a string, in a streamed chunk's choice or a body's choice. A value that is not null comes only in the choice that carries `finish_reason`: the chunk that finishes it, or a body's choice. trtllm-serve sets it there (`postprocess_handlers.py`, `output.stop_reason`) and leaves it out elsewhere |

Chat Completions has no standard mid-stream error. What trtllm-serve emits, and what a harness
does with it, are S0 measurements recorded in its contract (e.g. `contracts/pi.md`).

## 4. Rules common to all three

| Rule | Requirement | Check |
|---|---|---|
| WIRE-01 | A reply body is SSE or JSON. Anything else (an HTML error page, plain text) is broken | C-01 |
| WIRE-02 | A reply that the client did not cut has a body | C-04 |
| WIRE-03 | A request with `stream: true` is answered with a stream, unless it is an error (HTTP status 400 or above) | C-01 |
| WIRE-04 | A request without `stream: true` is answered with JSON, not a stream | C-01 |
| WIRE-05 | An event type, block type, output item type, choice or delta field that no rule in this file covers. It is recorded per element, and conformance is BLOCKED until a rule exists for it: the validator never passes what it did not check | C-11 |

- **The streamed bytes are the truth.** The client already received them, and may have acted
  on them. Every other view must agree with the stream, byte for byte.
- **Over-length, overload and unsupported fields are answered the way the harness can
  recover from.** The harness profile's `errors` section is the requirement, and checks C-10
  and R-03 are the test.
- **Unknown request fields are accepted, not rejected,** unless the harness profile says a
  rejection is the recovery signal (e.g. Claude Code retries without some capabilities only when
  the 400's wording matches). Main's Anthropic request models use `extra="allow"`.

## 5. A second reading by the official SDKs (check C-12)

Every rule above was written by hand from the protocol documents, and so was the adapter that
applies it. `scripts/validate/sdk_diff.py` gives the same recorded bytes to the vendors' own
Python SDKs instead: their SSE decoder, their event dispatch and their stream accumulators, as a
client built on them runs them. It uses none of the adapters' code.

| Protocol | Read with |
|---|---|
| Anthropic Messages | `anthropic` `BetaMessageStream` over `BetaRawMessageStreamEvent` (harnesses call `/v1/messages?beta=true`, e.g. Claude Code) |
| Chat Completions | `openai` `ChatCompletionStream`, its accumulated snapshot |
| Responses | `openai` `ResponseStream`: the `output_item.done` items, the snapshot it builds from `output_index`-addressed events, and the parsed `response.completed` |

The two readings are compared for every reply with a success status:

| Rule | Disagreement |
|---|---|
| SDK-01 | The SDK raised while reading the reply |
| SDK-02 | Other content: text, reasoning, calls (id, name, payload), signature presence, or the order of blocks or items. For Responses, also the SDK's index-addressed snapshot against the adapter's id-addressed items |
| SDK-03 | Another stop reason, finish reason or terminal |
| SDK-04 | Another message or response id, or model |
| SDK-05 | Other usage counts |

- **Any disagreement is FAIL.** Either the adapter misreads the wire, which is a validator
  defect, or the bytes admit two readings and a client built on the SDK acts on the other one.
  The investigator decides which, and a validator defect is fixed as a policy change.
- **An SDK that is not installed makes C-12 BLOCKED** for its protocols. The versions are pinned
  in `scripts/validate/requirements.txt`, recorded in `verdict.json`, and bound into the control
  receipt.
- **Not compared:**
  - replies with an error status, which C-10 and the error rules judge;
  - item and block types no rule covers, which are C-11's;
  - Responses reasoning text and custom tool input in the snapshot, which the SDK does not
    accumulate (the `output_item.done` items still carry both).
- **SDK behaviour that is not a finding:**
  - `ChatCompletionStream.get_final_completion()` is the structured-output helper and raises on
    every `length` or `content_filter` finish, so the accumulated snapshot is read instead;
  - an in-stream error event makes the SDK raise its own API error, which agrees with an adapter
    that read an error terminal.
- **A fault only the SDKs read becomes an adapter rule too.** Five faults that no rule caught
  were first read by C-12 alone, each through a fake-server control:
  - an Anthropic stream without `event:` lines;
  - a `response.created` without an `output` list;
  - message events carrying another item's `output_index`;
  - a first tool call announced at index 1;
  - later Chat fragments repeating the call's `id` and `name`.

  Each is now an adapter rule too (ANT-15, RSP-01, RSP-15, CHT-04), and the controls show both.
- **Limits.**
  - The Python SDKs are siblings of the TypeScript SDKs that Claude Code and pi build on, not
    the same code. Where they differ, the adapters keep the stricter rule. For example,
    openai-node assigns a repeated `id` where openai-python concatenates it, and the TypeScript
    Anthropic stream needs `message_stop` where the Python one does not (ANT-10).
  - Codex's own Rust client is not covered.

## 6. Usage (check C-14)

A wrong count does not break a stream. It makes the harness compact too early or overflow,
since harnesses size their context from it (e.g. Codex compacts on the reported total, Claude
Code sizes its context with the input count and with `count_tokens` before a turn, and pi keeps
a running total). `scripts/validate/usage.py`.

| Rule | Requirement |
|---|---|
| USG-01 | Required fields are present, integers, at least 0. Responses (`completed` or `incomplete`): `usage.input_tokens`, `output_tokens`, `total_tokens`. Messages: `input_tokens` and `output_tokens` after `message_delta` is merged in, plus `cache_read_input_tokens` and `cache_creation_input_tokens` when present. Chat: `prompt_tokens`, `completion_tokens`, `total_tokens` whenever usage is sent, and usage is sent when `stream_options.include_usage` asked for it or the body is JSON |
| USG-02 | A total is the sum of its parts: `total_tokens` = input + output (Responses), `prompt_tokens` + `completion_tokens` (Chat) |
| USG-03 | A detail is within its whole: cached tokens at most the input, reasoning tokens at most the output |
| USG-04 | A reply that delivered text, reasoning or a call reports at least one output token |
| USG-05 | `POST /v1/messages/count_tokens` answers the input the same request was billed: `input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens` of the `/v1/messages` turn whose prompt-shaping fields (model, messages, system, tools, tool_choice, thinking, output_config, context_management) are the same, compared without `cache_control` |
| USG-06 | The reported output equals the engine's own count of the generation (`num_output_tokens` of its raw record, with `--raw`) |
| USG-07 | The reported input equals the server's own prompt count in the trace record (`prompt_tokens.count`), less the prompt offset the model profile declares for the reply's protocol (`usage.prompt_offset`: one number, or one per protocol). A Chat request that ends its prompt itself (`chat_template`, `prompt_token_ids`, `prompt_token_ids_b64`, or `add_generation_prompt: false`) carries no offset |
| USG-08 | After a turn whose calls the harness echoed back exactly, the request that sends them back bills at least as much input: its prompt holds this turn. Not judged when that request has a shorter history (a compaction) or asks the server to clear history (`context_management`). When it declares strictly fewer of the turn's tools (its tool names a strict subset of the turn's, none included), it may bill less by at most the UTF-8 bytes of the turn's declarations it does not carry as they were, plus 512 tokens for the template's tools preamble once no tool is left (`validate.md`, C-14) |

- **Messages input.** trtllm-serve bills Anthropic input as the prompt minus cache reads and
  reports the cache reads separately (`anthropic_adapter.convert_usage`), so the whole prompt is
  the sum of the three input fields.
- **Prompt offsets are per protocol.** A model's serving extension may leave part of the
  rendered prompt out of the billed input (`num_prompt_tokens_offset`) on some routes only, so
  the model profile declares the offset per protocol (`usage.prompt_offset`), as the build it
  was measured on bills it. For example, Kimi-K3's extension leaves its 3-token generation opener
  out on Chat, and on Messages, which runs through the Chat handler, while the Responses route
  bills the whole prompt: `prompt_offset: {chat-completions: 3, anthropic-messages: 3,
  responses: 0}`.
  The offset applies to the prompt count (USG-07) only. `count_tokens` on main counts the whole
  rendered prompt, so with an offset on Messages it answers that much more than the turn billed,
  and USG-05, which holds the two equal, reports it: a client that budgets by `count_tokens` is
  then told another number than it is billed.
- **`count_tokens` is paired by content, not by id.** A smoke run can probe it: replay the
  recorded `/v1/messages` bodies' prompt-shaping fields to `/v1/messages/count_tokens` into the
  same trace, and C-14 pairs them.
