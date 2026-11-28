<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harness contract: Claude Code 2.1.289 against trtllm-serve

Harness profile: `profiles/harness/claude-code.yaml`. **Status: partial.** Wiring runs against
the fake server are recorded (section 9), and its timeouts are measured. The fields still marked
TODO in the profile are open.

Source marks:
- **[doc]**: Claude Code's gateway documentation
  (`code.claude.com/docs/en/llm-gateway-protocol` and `.../llm-gateway`);
- **[main]**: trtllm-serve on main;
- **[measure]**: an open S0 item.

Claude Code is closed source. Its semantics come from the docs plus black-box measurement:
record real sessions, then inject faults with the fake Messages server
(`references/acceptance-cases.md` section 2) and watch what it does.

**Scope warning [doc].** "Anthropic doesn't support routing Claude Code to non-Claude models
through any gateway." Capabilities arrive as new beta values, body fields and headers in new
releases. Pin the version and re-run this contract on every upgrade.

## 1. Identity

- **Version 2.1.289.** Record where the binary is installed, and pin it.
- **Point it at trtllm-serve with:**
  - `ANTHROPIC_BASE_URL=http://<host>:<port>` without `/v1`, because Claude Code appends
    `/v1/messages`;
  - `ANTHROPIC_AUTH_TOKEN=<dummy>` (sent as `Authorization: Bearer`);
  - `ANTHROPIC_MODEL=<served name>` [doc].
- **Per-session isolation:** set `CLAUDE_CONFIG_DIR=<dir>`. The model-discovery cache and
  settings live there [doc].
- **Recommended for smoke runs** [doc], each to be confirmed [measure]:

  | Variable | Why |
  |---|---|
  | `ANTHROPIC_DEFAULT_HAIKU_MODEL=<served name>` | Background tasks go to the main model anyway with a bearer credential; make it explicit |
  | `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` | Adds request class, compaction and prompt-id headers. Trace grouping needs them |
  | `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` | Fewer side calls outside the endpoint |
  | `CLAUDE_CODE_ATTRIBUTION_HEADER=0` | Removes the per-conversation attribution block from the system prompt. Decide per cell: it is part of what real users send |

## 2. Wire

- **[doc] Routes:**
  - `POST /v1/messages?beta=true`, always streamed;
  - `POST /v1/messages/count_tokens`, which is optional. Without it Claude Code estimates
    tokens from characters, and `/context` shows approximations;
  - `HEAD /api/hello`, a warm-up probe the endpoint may reject;
  - `GET /v1/models?limit=1000`, only with `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`. It
    keeps ids containing `claude` or `anthropic`, has a 3 s timeout, and treats a redirect as a
    failure.
- **[doc] Headers:**
  - `anthropic-version: 2023-06-01`;
  - `anthropic-beta`, an open, growing list;
  - `Authorization` and/or `x-api-key`;
  - `x-claude-code-session-id`, plus agent ids on sub-agent requests.
- **[doc] For a model id it does not recognize, Claude Code assumes a current Claude model and
  sends the full capability set:**
  - `thinking: {"type": "adaptive"}`;
  - `output_config` (effort, structured-output format, task budget);
  - `context_management` with its beta;
  - mid-conversation `role: "system"` entries in `messages`;
  - `cache_control` on system and message blocks;
  - beta tool fields (`strict`, `defer_loading`) and `tool_reference` blocks;
  - possibly the advisor tool.
- **[main] The request models use `extra="allow"`** and accept `thinking.adaptive`,
  `output_config`, `context_management` and `role: "system"` messages. The "rejected field"
  class (e.g. a 400 for a field Codex always sends, such as `service_tier`) is largely
  preempted. **[measure]** Confirm with real bodies.
- **[doc] System prompt.** Its first block is an attribution block: client version plus a
  fingerprint, stable per conversation since v2.1.181. On a non-Anthropic endpoint it reaches
  the model and the prefix cache.

## 3. Tools

- **[doc]** Tools are declared in `tools` with `input_schema`. MCP tool search adds
  `defer_loading` and `tool_reference`.
- **[measure]**
  - Which server tools (web search, advisor) reach the endpoint.
  - Whether it ever sends `tool_choice` `any`/`tool` or `disable_parallel_tool_use`. **[main]**
    rejects all three with a 400.
- **[measure] What it sends back for a failed or unknown call:** a `tool_result` with
  `is_error: true`, and content strings, e.g. `<tool_use_error>...`. These become the profile's
  `outcomes`.

## 4. Conversation state

- **History.** The full history is sent every turn [doc].
- **Reasoning round-trip.**
  - **[doc]** Thinking blocks come back with their `signature`. If the endpoint rejects a
    signature (e.g. "bound to a different conversation"), Claude Code drops earlier thinking
    from later requests.
  - **[main]** No signature is emitted, and incoming signatures are dropped.
  - **[measure]** What Claude Code does with unsigned thinking blocks. Then decide per model
    how past reasoning is rendered: model profile `reasoning.history_policy`.
- **[measured] Whitespace-only text.** A text block that is only whitespace is dropped when the
  turn is sent back, so its echo never matches. Serving should not emit one
  (`references/adapt.md` section 5).
- **Identity.**
  - Session: `x-claude-code-session-id`.
  - Sub-agents: `x-claude-code-agent-id` and `x-claude-code-parent-agent-id`.
  - Request class, compaction trigger and prompt id need hint headers [doc].

## 5. Stream semantics

- **[doc] Full event sequence.** Claude Code reads events as they arrive and expects the whole
  sequence, through `message_delta` and `message_stop`. A body that ends cleanly after a block
  has started, but before `message_delta`, is a dropped connection: it shows "the response above
  may be incomplete" or retries.
- **[doc] Idle watchdog.** Claude Code aborts a stream when no bytes arrive for longer than its
  streaming idle watchdog, and pings are the keepalive.
  - **[main]** No `ping` is emitted.
  - **[measure]** The watchdog value, and what happens on a long prefill. This is the analog of
    Codex's 120 s cut.
- **[doc] Retries.**
  - It honors `retry-after` (integer seconds); above 60 it stops retrying.
  - It reads `x-should-retry`.
  - It retries once without a capability when a 400's wording matches the capability: the
    `thinking` type, mid-conversation system messages, `cache_control` on a system message,
    a thinking signature, the advisor tool, `output_config.effort` and `context_management`.
  - The logic matches error wording, so **our error messages are part of this contract**.
  - **[measured, from the matchers in the 2.1.289 and 2.1.309 binaries]** The exact strings are
    in the harness profile's `errors.capability_retries`. The same in both versions, except
    `context_management: Extra inputs are not permitted`, matched since 2.1.304. Two cases are
    not wordings:
    - a request sent with `thinking.display: "updates"` is retried once without it after ANY 400
      that no other rule claims, and `display` is not sent again in that conversation. That is
      why main's "Unknown parameter: 'thinking.adaptive.display'" 400 is retried;
    - `display: "highlights"` is dropped to `omitted` on `thinking.(adaptive|enabled).display:
      Input should be`.
  - The binaries also carry a table of rejection classes for gateways (mid-conversation system,
    `cache_control`, thinking signature, thinking type, structured outputs, effort, media,
    prompt too long, `max_tokens` overflow, a rejected `anthropic-beta` value). Over-length is
    "Prompt is too long" or "Input is too long for requested model", at either status.

## 6. Context window

- **[doc]** For an unrecognized model id it assumes 200K, or 1M when the id carries `[1m]`.
  **[measure]** The model-config mechanism that declares the real window for a custom id. Set it
  to the served `max_seq_len`.
- **[doc] Reactive compaction.** It compacts when the API rejects a request as too long. Claude
  apps gateway uses the stable token `capability_rejected: prompt_too_long`.
  - **[main]** Over-length is an HTTP 400 with OpenAI wording and no "prompt is too long".
  - **[measure]** Whether Claude Code compacts on main's response. If it does not, this is
    Codex's over-length case again: an HTTP 400 where the harness needed another answer.

## 7. Known quirks and checks to enable

- **Background tasks.** Titles, classifiers and summaries go to the main model unless
  `ANTHROPIC_DEFAULT_HAIKU_MODEL` is set. They use structured outputs (`output_config.format`).
- **Direct egress.** The fast-mode check and the WebFetch domain safety check call
  `api.anthropic.com` directly. In a no-egress sandbox those errors are expected noise.
- **[observed] Output cap.** It sends `max_tokens: 32000` on every request, so the served
  `max_seq_len` must leave that much room above the prompt.
- **[measured] Opening prompt.** Version 2.1.289 opens with about 15.5k tokens. Without chunked
  prefill every prompt must fit in `max_num_tokens`, so turn chunked prefill on rather than raise
  `max_num_tokens` (`references/smoke.md` section 2).
- **A session's environment.** A Claude Code session exports `CLAUDE_CODE_*` variables,
  including a messaging socket and its token. A session started from inside another must not
  inherit them: run each under `env -i` (`scripts/run_claude_sessions.sh` does).
- **Checks:**
  - ANT-01 to ANT-14;
  - C-03 echo;
  - C-07 if parallel calls are ever disabled;
  - C-10 over-length;
  - R-01/R-02 against the measured watchdog;
  - H-01 to H-05 with the measured outcome strings.

## 8. Open S0 measurements (in order; struck items are answered in section 9)

1. Record 10 sessions: the session catalog in `references/contract.md` section 2, sub-agent and
   compaction included. Run `summarize_capture.py`. (Partly done: 5 wiring sessions, no
   sub-agent or compaction yet.)
2. Measure the idle watchdog and the first-byte timeout: use the fake server with a
   `silent_gap`.
3. Over-length: main's 400 vs `prompt is too long` wording. Which one triggers compaction?
4. Thinking round-trip: unsigned blocks, a rejected signature.
5. ~~Tool error strings: unknown tool, schema-invalid input, a failing command.~~ Done.
6. Whether `tool_choice`/`disable_parallel_tool_use` are ever sent.
6a. When Claude Code calls `count_tokens` (not in the 2-turn wiring runs). main's count includes
    Kimi-K3's 3-token generation opener that the billed input leaves out, so the K3 cell should
    expect USG-05 until one of them changes (`references/protocols.md` section 6).
7. The context-window declaration for a custom model id.

## 9. Measured: wiring runs against `fake_agent_server.py`

- **Setup.**
  - Claude Code 2.1.289 ran under `scripts/run_claude_sessions.sh`: `env -i`, its own
    `CLAUDE_CONFIG_DIR`, hint headers on, nonessential traffic off.
  - Allowed tools: Read, Glob, Grep. Bash, Write, Edit, NotebookEdit, WebFetch and WebSearch
    were disallowed.
  - Requests went through `record_requests.py` in forward mode.
  - Five sessions in all.
- **[measured] Routes.**
  - `HEAD /api/hello` first, then `POST /v1/messages?beta=true`.
  - Neither `count_tokens` nor `/v1/models` was called.
- **[measured] Body keys:** `model`, `messages`, `system` (3 blocks), `tools`, `max_tokens`,
  `stream: true`, `metadata`, `thinking`, `output_config`, `context_management`, and
  `safeguards` on the first request of a session. `safeguards` is not in the gateway guide's
  field list. main accepts it (`extra="allow"`); the validator reports it as an unknown field.
  - `thinking: {"type": "adaptive", "display": "updates"}`
  - `output_config: {"effort": "high"}`
  - `context_management: {"edits": [...]}`
  - `messages` contains mid-conversation `role: "system"` entries from the first request on.
- **[measured] Tools.**
  - 16 tools were declared once the six above were disallowed. They included Agent, Glob,
    Grep, Read, Skill, the Cron and Worktree tools, and SendMessage.
  - All were plain `input_schema` tools: no server tools, no `defer_loading`.
- **[measured] Headers:**
  - `anthropic-version`, `anthropic-beta`, `authorization`;
  - `x-claude-code-session-id`, `x-claude-code-request-class: main`, `x-claude-code-prompt-id`;
  - `x-claude-code-prev-tool-durations` on the request after a tool ran;
  - `anthropic-dangerous-direct-browser-access`.
  - `anthropic-beta` included `claude-code-20250219`, `interleaved-thinking-2025-05-14`,
    `context-management-2025-06-27`, `mid-conversation-system-2026-04-07`, `effort-2025-11-24`,
    `thinking-display-updates-2026...`, and others.
- **[measured] Echo.** The next request carried our `tool_use` back with the same id, name and
  input (check C-03 passed).
  - A second run (`ok_thinking`) delivered a signed thinking block, text and a call. The next
    request carried all three back verbatim and in order, signature included, as the assistant
    message appended right after the previous history: the profile's `echo` is `text: exact`,
    `reasoning: exact`, and C-03 compares them.
  - Claude Code re-serializes earlier messages between turns: the user message went out as
    text parts with `cache_control` and came back as a plain string. The validator's continuity
    digest ignores both differences (`turnlib.message_digest`).
- **[measured] Tool results.** All three kinds below have `is_error: true`:

  | Situation | Content |
  |---|---|
  | Unknown tool | `<tool_use_error>Error: No such tool available: NAME</tool_use_error>` |
  | Schema-invalid input | `<tool_use_error>InputValidationError: Read failed due to the following issue: The required parameter `file_path` is missing</tool_use_error>` |
  | The tool ran and failed | `File does not exist. Note: your current working directory is ...` (no wrapper) |

- **[measured] Fake-server trap.** A fake server that calls an unknown tool on every turn loops
  forever: Claude Code sent 3,013 requests in 240 s. The fake server now honors
  `--max-tool-turns` in every mode.
