<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# The turn record: one shape for three protocols

Validator-owned. Every check in `validate.md` reads turn records, never raw protocol JSON. One
adapter per protocol builds them, so a check is written once and runs on every harness's
traffic alike.

## 1. Inputs and identity

An adapter joins, per reply:
- the request record (`requests-*.jsonl`, or a recorder or replay capture);
- the response record (`responses-*.jsonl`, whose `response.body` is the raw SSE text or the
  JSON body);
- optionally the raw generation (`raw-*.jsonl` frames, on disaggregated deployments);
- optionally the **next request of the same thread**, which carries the harness's echo of our
  items and its outcome for every call.

Rules:
- **The cell comes from the run configuration, never from the body.** The `model` field is
  whatever the client sends: a harness, or a gateway in front of it, may send one fixed name
  whatever model serves it.
- **Content stays in memory.** A serialized turn record keeps lengths and SHA-256 only. Dumping
  content needs an explicit flag and an output directory with the traces' permissions.
- **Adapters are total.** Anything they cannot map is recorded, and nothing is dropped
  silently:
  - an event, block, item, choice or delta field without a rule is recorded per element
    (rule WIRE-05, check C-11) and makes conformance BLOCKED;
  - unknown request fields are listed in `unknown_fields`.

## 2. Schema

```text
TurnRecord
  ir: 1
  locator: {request, response, raw?, next?}    each {file, line, trace_id}
  protocol: responses | anthropic-messages | chat-completions
  route: str
  cell: {model, harness}                       from the run configuration
  identity:
    build_id: str?                             from the trace record (TRTLLM_BUILD_ID)
    session, thread, agent, parent_agent: str? keys come from the harness profile `identity`
    request_class: main | subagent | compaction | auxiliary | null
    turn_index: int?
  request:
    stream: bool
    model_field: str                           informative only
    declared_tools: [Tool]
    tool_choice: {mode: auto | none | required | named, name: str?}
    parallel_calls: allowed | single | unspecified
    reasoning: {requested: bool?, effort: str?, budget_tokens: int?}
    max_output_tokens: int?
    history: [HistoryItem]
    unknown_fields: [str]                      top-level keys outside the protocol schema (names only)
  reply:
    http_status: int
    transport: complete | client_disconnect | server_cut | unknown
    events: int
    first_event_s: float?                      needs per-frame timestamps
    max_gap_s: float?
    grammar: [Violation]                       rule ids from protocols.md
    views:
      stream: [Item]                           assembled from the deltas
      final: [Item]?                           see section 4
      final_na_reason: str?
      raw: {frames: int, text, stopped: partial | aborted | null}?   section 4
    terminal: {kind: completed | incomplete | failed | error | none,
               stop_reason: str?, error: {type: str?, code: str?, message_sha256: str?}?}
    usage: {input_tokens?, output_tokens?, cached_tokens?, reasoning_tokens?}?
  echo: [Item]?                                our items as the harness sent them back
  outcomes: {call_id: OutcomeClass}            from the next request's tool results

Tool
  name: str                                    bare name
  namespace: str?
  kind: function | custom | builtin | server
  declared_in: tools | additional_tools | namespace | other
  schema: object?                              JSON schema of the arguments
  grammar: {syntax, definition}?               custom tools (e.g. a Lark grammar)
  nested_names: [str]                          reachable only through this tool (harness profile)

Item
  kind: text | reasoning | call
  index: int                                   position in the reply
  id: str?                                     item id, block index or tool_call id
  call_id: str?
  name: str?                                   as delivered, possibly prefixed
  namespace: str?
  call_kind: function | custom | null
  payload: str?                                JSON arguments or custom input (memory only)
  text: str?                                   message or reasoning text (memory only)
  carrier: present | absent | null             encrypted_content / thinking signature
  status: str?

HistoryItem
  role: system | developer | user | assistant | tool
  kind: text | reasoning | call | tool_result | image | other
  call_id, name: str?
  is_error: bool?
  outcome: OutcomeClass?                       tool results only
  chars: int
  sha256: str

OutcomeClass: result | aborted | unsupported | arguments_rejected | unavailable_in_mode | error | pending
```

## 3. Mapping per protocol

| Field | Responses (`/v1/responses`) | Anthropic (`/v1/messages`) | Chat (`/v1/chat/completions`) |
|---|---|---|---|
| Declared tools | `tools` (incl. `namespace` groups), plus `additional_tools` input items; `custom` tools with `format.definition` | `tools[]` with `input_schema`. Server tools (`web_search_*`, advisor) are `server`. `defer_loading` is kept | `tools[].function` with `parameters` (and `strict`) |
| Tool choice | `"auto"`, `"none"`, `"required"`, `{type, name}` | `{type: auto \| any \| tool \| none, name}`; `any` maps to `required` | `"auto"`, `"none"`, `"required"`, `{type: function, function: {name}}` |
| Parallel | `parallel_tool_calls: false` gives `single` | `tool_choice.disable_parallel_tool_use: true` gives `single` | `parallel_tool_calls: false` gives `single` |
| Reasoning request | `reasoning.effort` | `thinking` (`enabled` + `budget_tokens`, `adaptive`, `disabled`), `output_config.effort` | `reasoning_effort`, `chat_template_kwargs` |
| History | `input` items: `message`, `function_call`, `custom_tool_call`, `*_call_output`, `reasoning` (with `encrypted_content`), `agent_message` | `messages[]` blocks: `text`, `image`, `tool_use`, `tool_result` (`is_error`), `thinking` (`signature`), `redacted_thinking`; mid-conversation `role: "system"` | `messages[]`: `system`/`developer`/`user`/`assistant` (`content`, `tool_calls`, `reasoning_content`) and `tool` (`tool_call_id`) |
| Stream items | `output_item.done` items, cross-checked against the deltas | One item per content block: `text`, `thinking` as reasoning (signature as carrier), `tool_use` as a call whose payload is the joined `partial_json` | `delta.content` as text; `delta.reasoning_content` or `delta.reasoning` as reasoning; `delta.tool_calls` by `index` as calls |
| Terminal | `response.completed` / `incomplete` / `failed` (`error.code`) | `message_delta.stop_reason`, then `message_stop`; `event: error` | `finish_reason`, then `[DONE]`; an error chunk |
| Echo (next turn) | `message`, `function_call`, `custom_tool_call` and `reasoning` items carrying our ids or call ids | `tool_use` blocks by id; text and thinking (with signature presence) from the `assistant` message at the position our reply was appended to, after the message it answered | `tool_calls` by id; `content` and `reasoning_content` of the `assistant` message at the appended position |
| Outcomes | `*_call_output.output` strings, by `call_id` | `tool_result` by `tool_use_id`: `is_error` and content strings | `role: tool` messages by `tool_call_id` |

**Which strings mean which outcome** is harness data (`profiles/harness/*.yaml`, `outcomes`),
never adapter code. Each harness writes its own (e.g. Codex writes `aborted`), and S0 measures
them.

## 4. Views and when one is not applicable

| View | Responses | Anthropic | Chat |
|---|---|---|---|
| Stream | yes | yes | yes |
| Streamed final snapshot | `response.completed.response.output` | none: `final_na_reason = "protocol has no streamed snapshot"` | none, same reason |
| Non-streamed body | `output` | `content` | `choices[].message` |
| Raw generation | `raw-*.jsonl` text, read in the model's own format (`native.py`), when the server recorded it | same | same |
| Echo | next turn | next turn | next turn |

- **For Anthropic and Chat streams, the echo is the cross-check.** It is also the view the
  harness acts on.
- **A view that should exist but is missing is BLOCKED.** A view the protocol does not have is
  NOT_APPLICABLE, with the reason above.
- **The raw view (C-13) is read without trtllm-serve's parsers.**
  - `scripts/validate/native.py` reads a raw record's text with the grammar of the checkpoint's
    own encoder: `kimi_xtml` (encoding_k3.py), `glm_xml` (chat_template.jinja), `dsml`
    (DeepSeek's encoding.py). The model profile's `native` section names the kind.
  - `tests/fixtures/native_goldens.json` holds renders made by those encoders
    (`scripts/validate/native_goldens.py`). Every one must read back exactly, and the DSML
    reader must agree with DeepSeek's own completion parser.
  - A raw record joins its reply by the ids the reply carries: the `disagg_request_id` the trace
    recorded, and the number in the reply's id, read as an engine id or a `disagg_request_id`. A
    Responses `resp_<id>_<hex>`, a Chat `chatcmpl-<id>` and an Anthropic `msg_<id>_<hex>` carry
    one, from a server that names its replies after the request (`validate.md`, C-13, has the
    order). An engine id that names several raw records (one per generation worker) joins the one
    that ran inside the reply's window with the reply's output count, or none (`validate.md`,
    C-13). A `disagg_request_id` that names two joins nothing.
  - A reading with anomalies (the model broke its own format) is the model's: it is flagged as
    M-07, and C-13 does not compare that reply (`not_judged`, outside its denominator). A slip
    that serving repairs and the model profile declares (`native.repairs`) is read the way
    serving reads it, and noted as `repair:<name>`.
  - A record says how its generation ended. The hook writes one when the request ends: at the
    final chunk (`aborted` set if the request was aborted), or, for a request that ended without
    one (an error), with `partial: true` and the request's `end_reason`. A partial or aborted
    record (`stopped`) holds what the model wrote until the server stopped it: the reply is
    compared as a prefix of it (RAW-05). A degenerate run in it is the model's (M-02), and fails
    R-04 only when the reply delivered it to the client.
  - The raw output itself is written by the raw-output hook
    (`tensorrt_llm/executor/raw_output_hook.py`), which main does not have yet. A trial build
    adds it as one of its capture changes (`smoke.md` sections 3c and 3d). Without it, C-13 is
    NOT_APPLICABLE.

## 5. Comparison rules

- **Arguments are compared as strictly typed JSON values.**
  - Duplicate keys, NaN and Infinity are rejected.
  - Key order and JSON spacing do not matter (lossless re-serialization).
  - Strings and numbers must match exactly, including their types.
- **Custom tool input and every text are compared byte for byte.** Whitespace is content.
- **A parser's corrected action is not the original action.** If serving repaired a name or
  payload, the turn record keeps both the raw and the delivered values.
- **Continuity is read by meaning, not by bytes.** Which earlier message a request continues is
  decided on a digest of each history message that ignores `cache_control` and reads a string
  content as one text part: harnesses re-serialize earlier messages between turns (e.g. Claude
  Code sends a user message as text parts with `cache_control` once, and as a plain string the
  next time).
