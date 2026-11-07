<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# S0: capture a harness's contract

Most serving bugs an agent harness hits are harness-contract bugs, not model bugs:
- a custom tool delivered as a function;
- a field the harness always sends being rejected;
- an error code the harness does not understand;
- 120 s of silence the harness reads as a dead server.

Write the contract first. Adaptation (S1) and the validator's harness profile are built from
it.

**Outputs:**
- `contracts/<harness>[-<config>].md`, with every line marked [measured], [source] or [doc];
- `profiles/harness/<id>.yaml`, derived per `contracts/TEMPLATE.md` section 8.

The human signs off once per harness version and configuration.

## 1. Point the harness at a recorder

- **Never capture against a vendor endpoint with real keys.** Use a self-hosted target.
- **Record-only** (learn the first request, before any server supports the harness):

  ```bash
  python scripts/record_requests.py --out-dir capture/ --port 8398
  # point the harness at http://127.0.0.1:8398 (Codex: .../v1), with a dummy key
  ```

- **Forward mode** (a server already answers, e.g. trtllm-serve with a small model or a fake
  server). It relays and records both sides byte-exact:

  ```bash
  python scripts/record_requests.py --out-dir capture/ --port 8398 --forward http://127.0.0.1:8000
  ```

- The recorder records the requests the server rejects too, with the server's answer.
- **The recorder records every route.** That covers `/v1/responses`, `/v1/messages` (plus
  `count_tokens`, `HEAD /api/hello`) and `/v1/chat/completions`.
- **Credentials.** The recorder redacts credential headers and forwards them only with
  `--forward-auth`. Bodies are recorded as sent, so treat captures like production traces.

## 2. Run a session catalog

Run sessions that exercise every path. At minimum:
- a plain question;
- one tool call;
- a forced multi-tool turn;
- a failing tool;
- a long tool output (or an image);
- an interrupt/cancel;
- an over-long context (paste until the harness complains);
- every reasoning-effort setting;
- a sub-agent, if the harness has them;
- a compaction, manual and automatic.

Each path is a field or event that otherwise shows up first in production.

## 3. Summarize the wire surface

```bash
python scripts/summarize_capture.py --out summary.json --markdown contract-skeleton.md capture/
```

- **What it reports:**
  - routes;
  - body keys with types and frequencies;
  - enumerated values;
  - every declared tool;
  - input item types, roles and content part types;
  - tool-output shapes;
  - metadata keys;
  - header names;
  - reply statuses, terminal events, error codes and the commonest SSE sequences.
- **What it never prints:** free text or argument values.
- **It understands Responses best.** For Anthropic and Chat captures, read message, tool and SSE
  shapes with the turn-record adapters (`proto_anthropic.py`, `proto_chat.py`; `validate.md`
  section 8).

## 4. Read the source (or probe) for what the wire cannot show

Answer each of these with file:line:

| Question | Example: Codex |
|---|---|
| Which event it records items from | `output_item.done` |
| Idle timeouts before and between events | About 120 s of silence and the stream is cut |
| Which errors it retries, and how many times | An HTTP 400 is terminal in the source, yet one deployment re-sent it six times: measure the harness as production runs it |
| What triggers compaction, and which error means "context full" | Only SSE `response.failed` with `context_length_exceeded` |
| How it matches a tool call to a declared tool | Bare vs namespaced names; custom vs function shape |
| What it replays every turn | Reasoning carriers, tool outputs |

- **Open-source harnesses** (e.g. Codex: codex-rs; pi: TypeScript): read the source.
- **Closed-source harnesses** (e.g. Claude Code): use the vendor docs, mark them [doc], then
  measure. Run the harness against the fake server of its protocol and inject one fault at a
  time:
  - silence for N seconds;
  - over-length errors of each wording;
  - unknown fields rejected;
  - signatures rejected.

  Record what it does: whether it retries, compacts, aborts, or drops a capability.

## 5. Write the contract and the profile

- **The contract.** Fill `contracts/TEMPLATE.md`.
  - `contracts/pi.md` is a measured example over pi's three APIs: tool results, errors, retries
    and timeouts against the fake servers' measurement modes (`acceptance-cases.md` section 3).
  - `contracts/codex-default.md` covers Codex with its default configuration.
  - `contracts/claude-code.md` lists its open measurements.
- **The profile.** Derive `profiles/harness/<id>.yaml`. Unknown fields stay `TODO`, which makes
  the checks that need them BLOCKED.
- **One profile per configuration.** The same harness version under another configuration is
  another contract. A configuration can move:
  - the tools from top-level `tools` into an input item (`additional_tools`);
  - the instructions from an `instructions` field into `developer` messages;
  - `parallel_tool_calls` from true to false.

  A harness that a hosted agent platform runs gets its own profile even under the same contract:
  e.g. Codex with its default configuration has `profiles/harness/codex-default.yaml` when run
  directly and `profiles/harness/codex-hosted.yaml` when a hosted platform runs it (its version is
  not visible there).

## 6. Traps

- **The model field is the client's, not yours.** A harness may send its own model id whatever
  model serves it. Route on pools, not on the model name. E.g. Claude Code sends whatever
  `ANTHROPIC_MODEL` says, and its model discovery keeps only ids containing `claude` or
  `anthropic`.
- **Tools may not be where you expect.** A configuration can send no top-level `tools` and
  declare them in an input item instead (an `additional_tools` item). A server or checker that
  only reads `tools` sees a tool-less request.
- **Namespaced tools are written both ways by models.** Decide, and test, which spellings the
  server resolves.
- **The session key is not the conversation.** For example:
  - Codex's `prompt_cache_key` is the parent session, and sub-agents share it: group by
    `client_metadata.thread_id`.
  - Claude Code puts sub-agents in `x-claude-code-agent-id` under one session id.
- **A wrapper changes the wire.** A proxy in front of the harness (e.g. LiteLLM) can add, drop or
  rename fields. Capture the harness the way production will run it.
- **Error wording can be part of the contract.** E.g. Claude Code's capability retries match the
  400's message text, and Codex compacts only on one error code.
- **Versions drift.** Re-run the capture when the harness version changes, and diff the
  summaries.
