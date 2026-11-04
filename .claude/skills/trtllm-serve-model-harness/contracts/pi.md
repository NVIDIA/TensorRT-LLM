<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harness contract: pi (coding agent) against trtllm-serve

Harness profiles: `profiles/harness/pi-chat.yaml`, `pi-responses.yaml` and `pi-messages.yaml`, one
per API. **Status: measured** (pi 1.4.0). Wiring was measured against the fake servers
(section 9) and against trtllm-serve main (section 10); outcomes, errors, retries and timeouts
against the fake servers (section 11). How pi's thinking level reaches each model's own effort
setting over Chat, and the model entry per model, are in section 12.

Source marks:
- **[doc]**: pi's README and `docs/models.md` (package `@mariozechner/pi-coding-agent` 0.68.0;
  the project now lives at `earendil-works/pi`);
- **[source]**: the installed `@earendil-works/*` 1.4.0 packages, `dist/` trees, file:line;
- **[main]**: trtllm-serve on main;
- **[measured]**: a run against a fake server or trtllm-serve, pi 1.4.0;
- **[measure]**: open.

pi is open source (TypeScript). Answer the [measure] items from its source with file:line, as
the Codex contract did with codex-rs.

## 1. Identity

- **Install and pin a version.** The package was renamed; record the exact package and version.
- **[doc] A custom provider lives in `~/.pi/agent/models.json`:**

  ```json
  {"providers": {"trtllm": {
     "baseUrl": "http://<host>:<port>/v1", "api": "openai-completions", "apiKey": "dummy",
     "models": [{"id": "<served name>", "name": "<label>", "reasoning": true,
                 "input": ["text"], "contextWindow": <served max_seq_len>, "maxTokens": <cap>,
                 "compat": { ... }, "thinkingLevelMap": { ... }}]}}}
  ```

  `thinkingLevelMap` sits beside `compat`, not in it; with `compat` it decides where pi's
  thinking level goes (section 12).

- **[doc] API choices:** `openai-completions` ("most compatible"), `openai-responses`,
  `anthropic-messages` and `google-generative-ai`.
  - **Use `openai-completions` for the matrix.** Codex already covers Responses and Claude Code
    covers Messages, so pi over Chat Completions covers the third path.
  - pi's other API choices can later serve as a protocol differential: the same agent, the same
    task, a different wire.
- **Isolation.** Use a separate `HOME` (or pi's own directory variable, [measure]) per session,
  so `~/.pi` is never shared.

## 2. Wire

- **[doc]** `POST <baseUrl>/chat/completions`.
- **[measure]**
  - Whether it always streams.
  - Which body keys it sends.
  - Whether it sends `stream_options.include_usage`, `parallel_tool_calls`, `reasoning_effort`
    and `chat_template_kwargs`.
- **[doc] Its `compat` flags are contract items.** Each one changes the wire:

  | Flag | Effect | trtllm-serve [main] |
  |---|---|---|
  | `supportsDeveloperRole` | `developer` vs `system` role | The developer role is passed through and may not render (the fix that renders `developer` messages instead of dropping them is not on main) |
  | `supportsReasoningEffort`; the model entry's `thinkingLevelMap` (1.4.0's name for `reasoningEffortMap`) | Sends `reasoning_effort` (the level name, or the map's string for it) | Read by K3's serving extension only (section 12) |
  | `thinkingFormat`, `chatTemplateKwargs` | `openai` sends `reasoning_effort`; `chat-template` sends `chat_template_kwargs` built from `chatTemplateKwargs`; `deepseek`, `zai`, `qwen`, ... their providers' fields | `chat-template` reaches GLM-5.3 and V4.1, `openai` reaches K3 (section 12) |
  | `maxTokensField` | `max_tokens` vs `max_completion_tokens` | [measure] |
  | `requiresToolResultName`, `requiresAssistantAfterToolResult` | Shape of tool results | [measure] |
  | `requiresThinkingAsText` | Past thinking is sent back as text | Interacts with the model's reasoning history policy |
  | `supportsUsageInStreaming` | `stream_options.include_usage` | Supported |
  | `supportsStrictMode` | `strict` in tool definitions | [measure] |

## 3. Tools

- **[doc]** The built-in tools are `read`, `write`, `edit` and `bash`; `grep`, `find` and `ls`
  are optional. They are declared as Chat `tools`.
- **[source] `edit`'s arguments are prepared before the call runs** (pi-coding-agent 1.4.0,
  `dist/core/tools/edit.js`, `prepareEditArguments`; pi-agent-core `agent-loop.js`,
  `prepareToolCallArguments`): `edits` given as a JSON string becomes what it parses to (an
  array, or one edit as `[edit]`), one edit object becomes `[edit]`, and a legacy top-level
  `oldText`/`newText` pair is appended to `edits`. The first two change the arguments object the
  history holds, so the next request sends the call back changed; the legacy fold makes a new
  object, which the history does not keep. In a GLM-5.3 smoke run over Chat, one call delivered
  `edits` as an object and came back as `[object]`. The pi profiles declare this as the echo
  rewrite and the tolerated repair `pi_edit_normalize` (C-03 `echo_rewritten`; M-01 `tolerated`
  when the prepared arguments are valid). `turnlib.pi_edit_stored` and `pi_edit_prepared` agree
  with pi's own function on 4,000 generated argument objects.
- **[source] Every call comes back as pi parsed it.** pi-ai 1.4.0 `dist/utils/json-parse.js`
  `parseToolCallArguments`, which each API runs on a finished call's arguments: JavaScript's
  `JSON.parse` (a duplicate key keeps its last value); failing that, the same after `repairJson`
  (a raw control character in a string escaped, a backslash that starts no valid escape
  doubled); `{}` with an `argumentsError` when the text is blank or still not a JSON object. The
  next request carries `JSON.stringify` of that object (`api/openai-completions.js`,
  `api/openai-responses-shared.js`) or the object itself (`api/anthropic-messages.js`). Arguments
  that are strict JSON come back as the same value; others do not: in a Kimi-K3 run, an `edit`
  call with a repeated key came back as the last value's JSON, shorter than delivered. The pi
  profiles declare it as the echo rewrite
  `pi_json_reserialize`; `turnlib.pi_tool_arguments` agrees with pi-ai's own on 3,017
  generated texts.
- **[measure]**
  - The error strings pi writes into `role: tool` messages for an unknown tool, invalid
    arguments and a failing command. These become the profile's `outcomes`.
  - Whether it ever disables parallel calls.

## 4. Conversation state

- **History.** The full history is sent every turn [measure].
- **Reasoning.** **[measure]** Which delta field it reads (`reasoning_content` or `reasoning`;
  trtllm-serve emits `reasoning_content`), and how it sends past reasoning back.
- **[measured] Whitespace-only text.** A text block that is only whitespace is dropped when the
  turn is sent back, so its echo never matches. Serving should not emit one
  (`references/adapt.md` section 5).
- **Session identity.** **[measure]** Whether any header or field identifies a session.
  - If none does, give each smoke session its own recorder port, so the capture tags the
    thread.
  - pi stores sessions as JSONL under `~/.pi/agent/sessions/` [doc]. That is a secondary,
    harness-side record.

## 5. Stream semantics

[measure]:
- the idle and first-byte timeouts;
- the retry policy;
- what it does with a mid-stream error chunk (Chat Completions has no standard one).

## 6. Context window

- **[doc]** The model entry's `contextWindow` (default 128000) and `maxTokens` (default 16384)
  drive pi. Compaction runs "on context overflow (recovers and retries)" and also proactively
  near the limit.
- **[measure]** How it recognizes overflow, i.e. which error status and wording. Then check
  that trtllm-serve's over-length answer triggers it (check H-04).

## 7. Checks to enable

- CHT-01 to CHT-12;
- C-03 echo;
- C-07 if parallel calls are disabled;
- C-09 reasoning separation;
- C-10;
- R-01/R-02;
- H-01 to H-04 with the measured strings.

## 8. Open S0 measurements (in order)

1. ~~Install and pin.~~ `@earendil-works/pi-coding-agent` 1.4.0 (section 9). Record 10 sessions
   over the session catalog. Run `summarize_capture.py`.
2. ~~Read the source for: the request builder (fields, compat flags), tool error strings,
   overflow detection, timeouts and retries.~~ Done and measured (section 11).
3. Pick `compat` values per model with trtllm-serve, then re-record. The profiles record pi's
   defaults for a custom base URL; a cell's model entry may change them. Effort: done, one model
   entry per model (section 12).
4. ~~Decide on session identity.~~ Responses carries a `session_id` header; Chat and Messages
   carry nothing, so the recorder tags each session (`record_requests.py --session`).
5. ~~Feed pi a Chat stream that ends with an unrequested usage chunk (`choices: []`).~~ pi always
   requests usage (`stream_options.include_usage`), so the K3 extension's chunk is a requested one
   for pi, and every run in section 11 accepted a usage chunk with `choices: []`.
6. ~~Echo.~~ Verbatim over all three APIs against the fake servers (section 9). Re-check over
   Messages against trtllm-serve, which sends no thinking signature.

## 9. Measured: wiring runs against the fake servers

- **Setup.** `@earendil-works/pi-coding-agent` 1.4.0 (renamed from `@mariozechner/pi-coding-agent`;
  node >= 22.19) under `scripts/run_pi_sessions.sh`, one session per API, `--tools read,ls`,
  behind `record_requests.py`. Chat and Messages against `fake_agent_server.py`, Responses
  against `fake_responses_server.py`.
- **[measured] Routes.** `POST /v1/chat/completions`, `POST /v1/responses` and
  `POST /v1/messages`, always `stream: true`. For anthropic-messages pi appends `/v1/messages`
  to the base URL, so that base URL has no `/v1` (with one, pi called `/v1/v1/messages`).
- **[measured] Bodies.**

  | API | Keys | Reasoning at the default level | Reasoning with `--thinking off` |
  |---|---|---|---|
  | openai-completions | `model`, `messages`, `tools`, `stream`, `stream_options.include_usage`, `max_completion_tokens`, `store`, `reasoning_effort` | `reasoning_effort: medium` | nothing |
  | openai-responses | `model`, `input`, `tools`, `stream`, `max_output_tokens`, `store`, `include`, `prompt_cache_key`, `reasoning` | `reasoning: {effort: medium, summary: auto}` | `reasoning: {effort: none}` |
  | anthropic-messages | `model`, `messages`, `system` (blocks), `tools`, `stream`, `max_tokens`, `thinking` | `thinking: {type: enabled, budget_tokens: 3072, display: summarized}` | `thinking: {type: disabled}` |

  With the model entry's `reasoning: false`, none of the three sends a reasoning parameter.
- **[measured] System prompt.** Sent as a `developer` message over Chat and Responses, as
  `system` blocks over Messages. It ends with the working directory (`<cwd>`).
- **[measured] Tools.** Function tools in each API's own shape. Over Messages, pi adds
  `eager_input_streaming`, `cache_control` on the last tool, and `"required": []` to a schema with
  no required parameter (`ls`).
- **[measured] Identity.** Responses: a `session_id` header and `prompt_cache_key`. Chat and
  Messages: none. Every API sends the `x-stainless-*` headers of the OpenAI or Anthropic
  TypeScript SDK.
- **[measured] Echo.** The next request carried our reply back verbatim: over Chat the assistant
  `content`, `reasoning_content` and `tool_calls`; over Responses the reasoning, message and
  function-call items with their ids; over Messages a signed thinking block, text and `tool_use`.
  C-03 passed on all three with `echo: exact` for text and reasoning.

## 10. Measured against trtllm-serve main (Qwen3-14B, one H200)

pi 1.4.0 ran four read-only tasks over each API against a main build
(`1.4.0rc2`, `--tool_parser qwen3 --reasoning_parser qwen3`), behind `record_requests.py`,
twice: comparable requests for the protocol differential, then pi's defaults.

- **openai-completions fails out of the box.** pi sends `store: false`; main's chat request
  model forbids unknown fields and answers HTTP 400 `extra_forbidden`. The body is
  `{"error": "<string>"}` (CHT-15) and the string carries a server source path and line.
  With the model entry's `compat.supportsStore: false`, every Chat check passed.
- **openai-responses.** Main's Responses stream fails RSP-07 (a `content_part.done` missing),
  RSP-12 (a reasoning item with `status: null` in the snapshot) and RSP-10 / C-02 (item ids
  differ between the stream and the snapshot). Look for existing fixes before writing new ones
  (`references/smoke.md` section 3b).
- **openai-responses renders an echoed turn as two assistant messages.** pi sends back a
  reasoning item and a function-call item; the Responses path renders them as two assistant
  messages (an extra `<|im_end|>\n<|im_start|>assistant\n`), the Chat path as one. The same
  history bills 5 to 6 more tokens on the Responses path (C-15 `server_render`).
- **anthropic-messages: `count_tokens` drops user text blocks.** pi (like Claude Code) sends user
  content as text blocks. The `count_tokens` renderer does not flatten them, so the user message
  renders empty and the count falls short of the billed input by the user text (10 to 17 tokens
  here, USG-05). The Messages route itself flattens them.
- **anthropic-messages: thinking comes back as text.** main sends no `signature_delta`; pi, with
  `allowEmptySignature: false`, replays the unsigned thinking block as a text block. The model then
  sees its own reasoning as visible text in the next turn (C-03 `text,reasoning`). Either the
  server signs thinking blocks or the model entry sets `allowEmptySignature: true`.
- **Schema key order is pi's.** Over Messages pi writes `type, properties, required`, over the
  OpenAI APIs `type, required, properties`; the server keeps the client's order, so that one-token
  difference is the harness's (`harness_literal`).
- **The official SDKs agree.** C-12 read every successful reply of the three APIs (26 per mode)
  with no disagreement.
- **Tool errors.** A failing read comes back as Node's error message,
  `ENOENT: no such file or directory, access '<path>'` (profile `outcomes.tool_error`).

## 11. Measured: outcomes, errors, retries and timeouts

- **Setup.** pi 1.4.0 on node 26.9.0 with its default settings (no `settings.json`), `-p --mode
  json`, tools `read,bash,edit,write`, one session per case under `env -i` with its own agent dir,
  behind `record_requests.py` (and once without it, for a raw cut). Chat and Messages against
  `fake_agent_server.py`, Responses against `fake_responses_server.py`, with their measurement
  modes (`references/acceptance-cases.md` section 3): `--fault-requests` and `--fault-skip` put a
  fault on chosen requests so recovery shows, `--arguments` makes a call fail. 52 cases, 11 of them
  timed (up to 660 s). The cases and scripts are `scripts/pi_kit/measure/`; their output stays
  outside the repository.
- **[measured] Tool results.** The same strings over all three APIs; over Messages every error
  result carries `is_error: true`, over Chat and Responses nothing marks one.

  | Situation | What pi sends back | Profile class |
  |---|---|---|
  | A call in a reply stopped at its token limit (`length`, `max_tokens`, `response.incomplete`) | `Tool call "<name>" was not executed: the response hit the output token limit, ...` | `aborted` |
  | A call to an undeclared tool | `Tool <name> not found` | `unsupported` |
  | Arguments that miss a required property | `Validation failed for tool "<name>":` and the schema errors | `arguments_rejected` |
  | Arguments that are not a JSON object | `Invalid arguments for tool "<name>": Tool arguments must be a JSON object ...` | `arguments_rejected` |
  | A command that failed | its output, then `Command exited with code <n>` | `tool_error` |
  | A file that does not exist | Node's error, `ENOENT: no such file or directory, ...` (section 10) | `tool_error` |

  **[source]** A run aborted mid-call answers `Operation aborted` (pi-agent-core
  `agent-loop.js:444`, `:509`, `:528`) and bash adds `Command aborted`; neither reaches the server
  under `-p`, which exits on abort. A `beforeToolCall` hook's block (`:514`) needs an extension.
- **[measured] Errors.** pi classifies every failure by its error text, not its status
  (**[source]** pi-ai `utils/retry.js` `isRetryableAssistantError`, `utils/overflow.js`
  `isContextOverflow`). Over each API:

  | Server answer | What pi does |
  |---|---|
  | HTTP 400 (a rejected field) | No retry; the run ends with the error. `-p` still exits 0, so read `stopReason` in the JSON events |
  | HTTP 429 | The identical request again after 2 s |
  | HTTP 500, every time | 4 identical requests, at 0, 2, 6 and 14 s; then the run ends with the error |
  | A stream that ends without its terminal event (no `finish_reason`, no `message_stop`, no terminal response event) | The identical request again after 2 s |
  | A stream cut at the socket | `terminated`; the identical request again after 2 s. A recorder in between ends its own reply cleanly, so pi sees the case above |
  | `finish_reason` without `[DONE]` | Accepted |
  | Over-length after three tool turns: a 400 in OpenAI's envelope, in trtllm-serve's top-level body (`{"object": "error", "message", ...}`), in Anthropic's (`prompt is too long`) or with OpenAI's wording over Messages, and an in-stream `response.failed` with `context_length_exceeded` | Compaction: a summary request without tools, then the turn sent again from the summary and the kept recent turns. One compact-and-retry per run (`agent-session.js:2445`) |
  | Over-length on the first request | Nothing to compact: the run ends with the error |

  Retries: 3 at most, after 2, 4 and 8 s (`retry.maxRetries`, `retry.baseDelayMs`); `retry-after`
  is not read, because provider-level retries, which read it, are off. pi drops no field after a 400.
- **[source] What counts as over-length or transient.** Over-length: the 26 patterns of
  `overflow.js:36-65` (the profiles carry them, `errors.overlength.matches`), unless a rate-limit
  pattern matches (`:76-80`); also a completed reply whose usage input exceeds the model entry's
  `contextWindow`, and a final reply stopped at `length` with less output than `maxTokens`, which is
  compacted and retried once. Transient: `retry.js:26-106`, including the status codes 429, 500,
  502, 503, 504, 520 and 524 anywhere in the text, `internal error`, `timeout` and `terminated`.
  So a 400 whose message says "Internal server error" is retried as if transient, and H-02 counts
  those resends against the server.
- **[main, checked at 4969c916f3] trtllm-serve's over-length wording is one pi reads.**
  A prompt that leaves no room for one generated token is refused with OpenAI's wording,
  `This model's maximum context length is M tokens. However, ...` (`executor/utils.py`
  `context_length_exceeded_message`, raised from `base_worker.py` when the deduced
  `default_max_tokens` is not positive, code `context_length_exceeded`). Two of pi's patterns match
  it (`maximum context length is \d+ tokens`, `reduce the length of the messages`), so pi compacts.
  Older builds said `` `default_max_tokens` (N) must be greater than 0 ``, and with chunked
  prefill off main still refuses a prompt past `max_num_tokens` with `The prompt length (N) should
  not exceed max_num_tokens (M)` (`llmapi/llm.py` `_check_arguments`). pi reads neither as
  over-length nor transient: there pi ends the run where it should compact. C-10 judges it on a
  live run (`errors.overlength.shape: any_error`).
- **[measured] Timeouts.** One undici dispatcher with `headersTimeout` and `bodyTimeout` from
  `httpIdleTimeoutMs`, 300 s by default (`core/http-dispatcher.js:77`, `:81`), and the SDKs'
  timeout, which ends when the headers arrive (`core/sdk.js:200`):

  | Fault | Over each API |
  |---|---|
  | No headers for 330 s | Cut at 299.99 s (`Request timed out.`), resent after 2 s |
  | Silence for 330 s after the first event | Cut at 300.7 s (`terminated`), resent after 2 s |
  | Silence filled with a `ping` (Messages) or an SSE comment (Chat, Responses) every 5 s | Held open for 330 s; for 660 s over Chat and Messages. No total cap |

  `httpIdleTimeoutMs: 0` in `settings.json` disables both timers.

## 12. Thinking level and each model's effort

With pi's defaults for a custom base URL, pi's default level goes out as `reasoning_effort:
medium` and reaches no model: GLM-5.3 renders `Reasoning Effort: Max` on every prompt and Kimi-K3
`thinking_effort=max`. A V4.1 entry with `chatTemplateKwargs: {thinking}` only sends no effort,
so DeepSeek V4.1 renders its default 75.

- **[source] pi's levels.** `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; the default
  is `medium` (pi-coding-agent `core/defaults.js:1`; `settings.json` `defaultThinkingLevel`
  changes it, `--thinking` overrides it). The level is clamped to the model entry (pi-ai
  `models.js:693-723`): a level its `thinkingLevelMap` maps to `null` is not offered, `xhigh` and
  `max` are offered only when the map names them, and a level that is not offered becomes the
  next offered one up, else down. Without a map, `--thinking xhigh` and `--thinking max` both
  send `high`. At `off` pi passes no level (pi-agent-core `agent.js:310`).
- **[source] Where the level goes over openai-completions** (pi-ai `api/openai-completions.js`
  `buildParams`, :623-756): one branch per request, chosen by `compat.thinkingFormat`.
  - `openai`, the default for a custom base URL (`detectCompat`, :1281-1291): top-level
    `reasoning_effort`, the map's string for the level or else the level name, when
    `supportsReasoningEffort` (default true) (:718-721). At `off`, the map's `off` string, else
    nothing (:722-727).
  - `chat-template`: `chat_template_kwargs` built from `compat.chatTemplateKwargs` (:651-656,
    :772-798). Plain values go as they are; pi fills `{"$var": ...}`: `thinking.enabled` (true
    unless `off`), `thinking.effort` (the map's string for the level or else the level name; a
    `null` is dropped; at `off` the map's `off` or nothing, and `omitWhenOff: true` drops it at
    `off`) and `thinking.budget` (pi's token budget, 1024, 2048, 8192 and 16384 from `minimal` to
    `high`, `xhigh` and `max` as `high`; `api/simple-options.js:46-58`). No top-level
    `reasoning_effort` in this format.
  - `deepseek`, `zai`, `qwen`, `qwen-chat-template`, `together`, `openrouter`, `baseten`,
    `string-thinking`, `ant-ling`: their providers' fields.
  - `thinkingLevelMap` values are strings or `null` (`providers/model-schema.js:11-14`), so pi
    sends no number through either path. `samplingParamsByThinkingLevel`, another model entry
    field (pi-coding-agent `core/model-config.js:47`), merges any body fields per level, last and
    shallow (:751-755): the only way to send a number, and it replaces a whole
    `chat_template_kwargs`. None of the three models needs it.
- **[main] What reaches each model over Chat** (main at c67c0f4fbe). The route runs the model's
  serving extension (`serve/render/chat.py:70-92`) and hands the request's `chat_template_kwargs`
  to the template or custom tokenizer (`openai_server.py:2381`, `inputs/utils.py:722-835`). The
  top-level `reasoning_effort` reaches only gpt-oss (Harmony) and K3's extension. The Chat request
  model takes `low`, `medium`, `high`, `max` and `none`, and answers 400 to `minimal` and `xhigh`
  (`openai_protocol.py:1129`).

  | Model | Its knob | The field that reaches it | What it accepts |
  |---|---|---|---|
  | GLM-5.3 (Jinja template, first lines) | `<\|system\|>Reasoning Effort: Low\|High\|Max` | `chat_template_kwargs.reasoning_effort`; the unused-kwargs guard passes it because the template reads it (`inputs/chat_template_guard.py:121`) | `low`, `high`; anything else, or nothing, renders Max |
  | Kimi-K3 (`encoding_k3.py`, a `thinking-effort` system message) | `thinking_effort=low\|high\|max` | top-level `reasoning_effort`, which `serve/extensions/kimi_k3.py:166-215` turns into `chat_template_kwargs.thinking_effort` (`none` turns thinking off, `medium` is dropped); also `thinking.effort` and `chat_template_kwargs.thinking_effort` | `low`, `high`, `max`, default `max` (`tokenization_kimi.py:379`); any other `thinking_effort` fails the encoder's assertion (400) |
  | DeepSeek V4.1 (`tokenizer/deepseek_v41`, `--custom_tokenizer deepseek_v41`) | `Reasoning Effort: N (range 1-100, ...)`, thinking mode only | `chat_template_kwargs.reasoning_effort`, with `chat_template_kwargs.thinking` (or `enable_thinking`) true | an integer 1-100; `low`, `high`, `max` (50, 75, 100); OpenAI's names between them, `minimal` 25, `medium` 62, `xhigh` 87 (`tokenizer.py:92-103`); nothing renders 75; `"75"`, 0 and 101 are rejected (400) |

  No server change is needed: each knob is reachable through pi's own fields.
- **The model entries** (the pi kit's presets, `scripts/pi_kit/presets/*.json`, `pikit run
  --compat-preset NAME`). Thinking stays on as before; pi's levels map onto each model's by
  meaning and in order. Build-dependent flags still go on top (`supportsStore: false` before
  #20864, `supportsDeveloperRole: false` for GLM-5.3 on a build without the fix that renders
  `developer` messages).

  | Model (preset) | `compat` | `thinkingLevelMap` |
  |---|---|---|
  | GLM-5.3 (`glm-5.3`) | `{"thinkingFormat": "chat-template", "chatTemplateKwargs": {"reasoning_effort": {"$var": "thinking.effort"}}}` | `{"off": null, "minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}` |
  | Kimi-K3 (`kimi-k3`) | `{"thinkingFormat": "openai", "supportsReasoningEffort": true}` | the same as GLM-5.3 |
  | DeepSeek V4.1 (`deepseek-v41`) | `{"thinkingFormat": "chat-template", "chatTemplateKwargs": {"thinking": {"$var": "thinking.enabled"}, "reasoning_effort": {"$var": "thinking.effort", "omitWhenOff": true}}}` | `{"minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}` |

  | pi level | GLM-5.3 | Kimi-K3 | DeepSeek V4.1 |
  |---|---|---|---|
  | `off` | not offered (it always thinks): runs as `minimal` | not offered: runs as `minimal` | chat mode, no effort line |
  | `minimal` | Low | low | 25 |
  | `low` | Low | low | 50 |
  | `medium` (pi's default) | High | high | 62 |
  | `high` | High | high | 75 (its default) |
  | `xhigh` | Max | max | 87 |
  | `max` | Max | max | 100 |

  - Why: GLM-5.3 and K3 have three levels and always think: pi's two lightest levels go to
    `low`, `medium` and `high` to `high`, `xhigh` and `max` to `max`. `off` is marked unsupported
    instead of left unmapped, because an unmapped `off` sends nothing and both models then run at
    their maximum. V4.1's budget is continuous (its README: 1-100, intermediate values
    interpolate) and the served tokenizer already places pi's level names on it, so they travel
    as they are; `off` keeps V4.1's chat mode.
  - With these entries, pi's default, `medium`, runs GLM-5.3 at High, K3 at high and V4.1 at 62.
    The model cards evaluate at the maximum: `--thinking max` (GLM-5.3 Max, K3 max, V4.1 100). To
    match runs made with pi's defaults (GLM-5.3 Max, K3 max, V4.1 75), use `--thinking max`,
    `max` and `high`.
  - The GLM entry is for templates that read `low` and `high` (GLM-5.3, GLM-5.3-Flash): GLM-5.2's
    knows only `high`, and a template that does not read `reasoning_effort` makes the guard answer
    400.
- **[measured, offline] Evidence.** pi 1.4.0 against `fake_agent_server.py` (Chat, `ok`)
  through `record_requests.py`, each preset and each configuration without one, at pi's default
  and at each level: 48 sessions, 96 requests, every session exited 0. Every request
  carried the field and value above; without presets, `reasoning_effort` with the level name
  (`xhigh` and `max` sent as `high`, nothing at `off`), and for V4.1 only `thinking`. Every
  recorded body then went through main's chat path on the CPU (`ChatCompletionRequest`,
  `prepare_chat_request`, `parse_chat_messages_coroutines`, `render_conversation` with the served
  tokenizers and GLM-5.3-NVFP4's template; modules that need the build stubbed): every prompt
  carried the marker in the table. Without presets GLM-5.3 rendered Max at every level, K3 max at
  `off`, `medium` and the default, V4.1 75 at every thinking level; and `--thinking minimal` sent
  `reasoning_effort: minimal`, which the Chat request model answers with a 400 for GLM-5.3 and
  K3, so pi ends the run. 40 synthetic requests through the same path give the "accepts" column.
  The evidence stays outside the repository.
- **Other APIs** (not checked here). Over `openai-responses` pi sends `reasoning.effort` (the map
  applies), which the Responses route copies into `chat_template_kwargs.reasoning_effort` with
  `thinking: true` (`responses_utils.py:1621`). Over `anthropic-messages` it sends `thinking` with
  a token budget, and `output_config.effort`, which the Messages route maps, only with
  `compat.forceAdaptiveThinking`. The presets are for `openai-completions`.
