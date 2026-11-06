<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex, default configuration: pointing it at trtllm-serve

Harness profile: `profiles/harness/codex-default.yaml`.

Checked against codex-cli 0.155.1. Re-check `codex exec --help` for other versions.

## Provider configuration

Codex talks to a custom provider through `model_providers`. These overrides need no config file
(`scripts/run_codex_sessions.sh` passes them):

```bash
codex exec --json --skip-git-repo-check -C <workdir> -s workspace-write -m <model> \
  -c 'model_provider="trtllm"' \
  -c 'model_providers.trtllm={name="trtllm-serve", base_url="http://<host>:<port>/v1", wire_api="responses", env_key="TRTLLM_API_KEY"}' \
  -o last_message.txt "<task prompt>"
```

From a script, give `codex exec` its input with `< /dev/null`: with a standard input that is not
a terminal, it waits to read more of the prompt from it (`scripts/run_codex_sessions.sh` does this).

| Setting | Notes |
|---|---|
| `wire_api="responses"` | Codex then POSTs `/v1/responses` with `stream: true` |
| `env_key` | Names the environment variable holding the key. Set it to a dummy when the endpoint needs no auth |
| `model_context_window` | Codex compacts when its own context count reaches this. Set it to the served window, or to a ceiling when compaction is off |
| `model_auto_compact_token_limit` | Related compaction setting |
| `model_reasoning_effort` | Codex otherwise sends whatever its config has. With no value, the endpoint's default applies |
| `CODEX_HOME=<dir>` | Per run, so nothing reads or writes the user's `~/.codex`. Codex warns that it cannot create helper binaries under `/tmp`; that is harmless |
| `-m <model>` | A name Codex does not know logs "Model metadata for `<name>` not found" in every session, and Codex falls back to defaults. Harmless: the declared tools were the same for every name tried, and `model_context_window` gives it the window |

## What changes with the configuration

The same Codex sends a different contract depending on its configuration: which tools it
declares and how (top-level JSON functions, or custom tools with a grammar), where the
instructions go, `parallel_tool_calls`, the reasoning request and the model id. Capture each
configuration with `scripts/record_requests.py` (see `references/contract.md`) and give it its own
harness profile. codex-cli 0.155.1 with the default configuration sends:

| Field | Default config |
|---|---|
| Tools | Top-level `tools`: `exec_command`, `write_stdin`, `request_user_input`, `view_image`, `multi_agent_v1.*`, goal tools, `web_search` (all JSON functions) |
| Instructions | `instructions` field |
| `parallel_tool_calls` | `true` |
| Reasoning | `reasoning.summary: "auto"` |
| Echo | **[measured]** The next request carries each reply back unchanged: its text, its reasoning items with their `encrypted_content`, and its calls (codex-cli 0.155.1 against `fake_responses_server.py`: 12 sessions, 24 calls, 24 texts and 24 reasoning items, all equal) |
| Model id | as configured |

Smoke-test the configuration that production will use. A pass on one says little about another:
e.g. a custom `exec` tool with a grammar and a JSON `exec_command` exercise different parser paths.

## The same contract as a hosted agent platform runs it

Profile: `profiles/harness/codex-hosted.yaml`. A hosted agent platform that runs Codex with its
default configuration against your endpoint (Responses API) sends the default-config contract
above. Measured from the first requests of a run on such a platform:

- Top-level `tools`: `exec_command`, `write_stdin`, `request_user_input`, `view_image`, the goal
  tools, a `multi_agent_v1` namespace and `web_search`. No `additional_tools` item.
- An `instructions` field; `parallel_tool_calls: true` (except on the compaction request, below);
  `tool_choice: "auto"`; `store: false`; `include: ["reasoning.encrypted_content"]`;
  `reasoning: {"summary": "auto"}` with no effort, so the server's default effort applies (and a
  reasoning object is what turns thinking on).
- `model` is the model id the platform was given for your endpoint. `client_metadata` carries
  `session_id`, `thread_id`, `turn_id`, `root_turn_id`, `x-codex-installation-id`,
  `x-codex-turn-metadata` and `x-codex-window-id`.
- **Compaction.** When its context count reaches `model_context_window`, Codex asks for a summary
  with a request of its own: the previous request's whole input, the turn's reply items and call
  outputs, then one appended user message (the compaction prompt); `tools: []`,
  `parallel_tool_calls: false`, the same `instructions`. The next request starts over (developer
  and user messages, the tools again) and carries the summary at the end of its last user message.
  A call in the summary reply is dropped, never answered. So the compaction request can bill less
  than the turn before it although it holds that turn: it no longer declares the tools, and USG-08
  allows a shrink up to what their declarations take (`references/validate.md`, C-14). The summary
  reply is not echoed by design (C-03 `*_not_echoed_compacted`).

What the platform itself changes:

- The requests come through the platform's gateway. It adds its own headers (its run ids, the
  model, a request id), forwards `client_metadata` as the only extension field, and sends no
  `user-agent`, so the Codex version is not visible. Record what arrives at the server, not what
  Codex writes.
- The endpoint must be one the gateway is allowed to reach.
- Agents print their environment (`env`, `curl -v`), so the platform's credentials can appear in
  tool outputs. Treat traces as secret and scan them for credentials
  (`scripts/validate/scan_credentials.py`).

## Over-length on main

Codex reads an over-length answer only as an in-stream failure with code
`context_length_exceeded` (the profile's `errors.overlength`, C-10), which the profiles list as a
compaction trigger next to reaching `model_context_window`. An HTTP 400 is resent as it was and then
ends the session. Against `fake_responses_server.py`, which reports tiny token counts, codex-cli 0.155.1
ended the session after either answer without compacting, so measure compaction (H-04) against a
real server.

On main, a streamed `/v1/responses` request over the window is answered with an HTTP 400, or
in-stream with code `server_error`. PR #21060 answers it in-stream with
`context_length_exceeded`; add it to the trial build of a Codex cell until it merges
(`references/smoke.md` section 3d). A smoke run at the full window rarely gets there, so probe it
on purpose (`references/smoke.md` section 2, step 4).

## Trial build items (2026-11)

The GLM-5.3 and Kimi-K3 Codex cells passed on main plus, besides the capture changes and the
model's own items (its profile's `upstream`):
- #21060: over-length answered in-stream (above);
- #21058: reasoning round-trips through `encrypted_content`, which Codex sends back on every turn.

#20868 (the final snapshot keeps the status the stream reported) and #21065 (a `web_search` tool
the server cannot honour is served, not refused) were in those builds and have merged since.
Check each item's state before composing a build (`references/smoke.md` section 3d).
