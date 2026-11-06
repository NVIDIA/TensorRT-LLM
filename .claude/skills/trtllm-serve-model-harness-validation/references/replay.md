<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# S2/S3: replay recorded agent traffic

An agent harness resends its instructions, tools and the whole history on every turn, so every
recorded request is a complete prompt. That makes traces a reusable test bed at three levels.

| Level | Input | Runs | Proves | Cannot prove |
|---|---|---|---|---|
| L1 frame replay | Raw generation frames | Real parsers and event pipeline, CPU only | Parser/stream determinism; stream == final for the recorded generations | Anything about a model with no recorded frames |
| L2 fake engine | Recorded token ids or text | The real HTTP handler with a fake engine | Event sequences, ids, terminals, error mapping, keepalive | Model behavior |
| L3 request replay | Recorded request bodies | A live endpoint (new build or new model) | Real harness prompts produce well-formed, consistent, valid calls on this build and model | Trajectories: tools are not executed, so a session cannot continue |

**S2** runs L1 and L2 on every model already supported after any change to a shared layer.
**S3** runs L3 against the new model. Hand everything to the validator (`validate.md`), exactly
like production traffic.

## 1. What the traces hold

Two writers make a trace:

| Writer | File | Contents |
|---|---|---|
| `scripts/record_requests.py`, in front of the server (`references/contract.md`) | `requests-capture.jsonl` | One line per request: `body` as sent, `headers`, `route`, `trace_id`, `build_id` |
| | `responses-capture.jsonl` | One line per reply: `response.kind = sse_text` with the SSE text, byte-exact; `status`; `timing` with per-chunk arrival; `build_id` |
| The raw-output hook (`TRTLLM_RAW_OUTPUT_DIR`, `references/smoke.md` section 3d) | `<UTC hour>/raw-<pid>.jsonl` | One line per generation: `request_id`, `disagg_request_id`, `frames[]` (the real chunk boundaries), `text`, `token_ids`, `streaming`, `aborted`, `output_index` |

`replay_requests.py` writes the same request and reply shapes. The readers also take the trace of
a server-side trace writer (`TRTLLM_REQUEST_TRACE_DIR`) that is not on main.

- **Joins.**
  - Request ↔ response on `trace_id`.
  - Response ↔ raw on the engine request id the reply carries, or on `disagg_request_id`,
    compared as an exact decimal string (the values exceed 2^53).
- **Raw frames may sit in more than one place,** e.g. in the per-attempt directories of a
  deployment that restarted. Look in all of them before calling a reply unjoined.
- **Handle traces as sensitive.**
  - Agents print credentials into tool outputs, and models copy them into their own output.
  - Never print record content.
  - Never copy unscreened lines into a corpus, a bug report or a chat.
- **A harness with no recorded traffic has no corpus** until its L4 runs (`smoke.md`) produce
  traffic.

## 2. L1: frame replay (CPU only)

- **What it does.** It feeds each raw record's `frames` through the real serving modules (the
  event generator with the chosen parsers, the rest stubbed) and fails on any reply whose `done`
  payload differs from its deltas.
- **Tool.** No frame replayer ships with this skill. Cover the same ground with:
  - `scripts/chunk_equivalence.py`: each raw record's `text` (one `{"id", "text"}` line of
    `--samples` per record) through the real tool parser, split at every point that matters,
    stream against whole;
  - L2 below: the real handler and event generator at real token boundaries.
- **When.** After any change to `responses_utils.py`, the Chat/Messages streaming path or a
  parser, over every model's recorded frames.
- **New models.** A new model has no frames until it has served. Collect its first raw output in
  L3 or L4, and add a representative set to the regression corpus. From then on, parser changes
  can be iterated on CPU against those frames. GPU time is only needed to sample new behavior.
  The renderer must be final before sampling.

## 3. L2: fake engine through the real handler

The pattern:
- build the server with `object.__new__(OpenAIServer)`;
- use an engine that returns recorded or rendered token ids through the real tokenizer's
  incremental decoding;
- POST a recorded body to the real handler;
- compare the emitted SSE with the recorded SSE, or run the checks on it.

On main, `_server_streaming` in `tests/unittest/llmapi/apps/test_responses_streaming_events.py`
builds the server the same way, with a generator that streams text.

Use it to pin a protocol bug to a commit once L1 or L3 has found it, and to confirm parser-level
findings at real token boundaries before filing them.

## 4. L3: per-turn request replay

```bash
python scripts/build_replay_corpus.py --workers 16 --out corpus.jsonl --manifest corpus.manifest.json \
    --per-thread 3 --max-records 2000 --model <served-model-name> --reasoning drop \
    --max-output-tokens 4096 --route <harness route> <request shards or hour dirs>
python scripts/replay_requests.py --base-url http://<host>:<port> --out-dir replay_out \
    --concurrency 16 --first-event-timeout <first_event_s> --build-id "$BUILD_ID" corpus.jsonl
python scripts/validate/verdict.py --harness profiles/harness/<h>.yaml --model profiles/model/<m>.yaml \
    --matrix <run>/matrix.yaml --population replay --build-id "$BUILD_ID" \
    --out-dir replay_out/verdict replay_out
```

- `<harness route>` is the route the harness's requests used: `/v1/responses` (Codex),
  `/v1/messages` (Claude Code) or `/v1/chat/completions`, or for pi the API it ran over. The
  builder keeps only requests on that route, and its default is `/v1/responses`.
- `<first_event_s>` is the harness profile's `timeouts.first_event_s` (Codex 120, pi 300, Claude
  Code 360): a replay that gives up sooner than the harness would records a timeout the harness
  never had.
- `$BUILD_ID` is the identity of the build that serves the replay, read on the serving host as for
  the recorder (`smoke.md` section 2: the `pip show tensorrt_llm` version with its commit). The
  replay writes it into every record and the verdict checks them against it. Without it nothing
  shows which build answered, so that inventory result does not count for the stage.

### Selection

- **Group by thread, not by request.** Turns of one session share almost the whole prompt.
- **The thread key comes from the harness profile's `identity`.** For example:
  - Codex: `client_metadata.thread_id`. Its `prompt_cache_key` is the parent session, shared by
    sub-agents.
  - Claude Code: `x-claude-code-session-id` plus the agent id.
- **`--per-thread 3 --pick spread`** takes the first, a middle and the last turn: short, medium
  and long contexts.

### Screens

| Screen | Why the record is dropped |
|---|---|
| Strong credential signatures | Never copied |
| Degenerate context | Input reasoning ending in a degenerate token (`--degenerate-suffix`, default `[PAD]`): a runaway the client replays every turn |
| `previous_response_id` | The body is not self-contained |
| Non-accepted requests | Not replayable |

The manifest counts every drop.

### Transforms

- **`--reasoning drop` (default).** Reasoning recorded from another model must not be fed to the
  new one as its own. `strip-encrypted` keeps only visible text; `keep` is for same-model
  replays.
- **`--model`.** Rewrite the model name for the endpoint under test.
- **`--max-output-tokens`.** Bound cost.

### Per-protocol handling

- **Thread keys.** Claude Code threads are keyed by `x-claude-code-session-id`, plus the agent id
  for sub-agents.
- **`--reasoning drop`.** It also removes Anthropic `thinking`/`redacted_thinking` blocks and
  Chat `reasoning_content`/`reasoning`. `strip-encrypted` drops thinking signatures.
- **The output cap** goes to the protocol's own field: `max_output_tokens`, `max_tokens`, or
  `max_completion_tokens`.
- **`/v1/messages` replays** carry `anthropic-version` and, with a key, `x-api-key`.
- **Timing.** Replay records per-chunk timing, so R-01/R-02 can be judged.

### Reading the result

- **Judge mechanically, never by content.** Outputs are not deterministic.
  - Every reply terminated.
  - Views agree.
  - Each call is valid for its own request's tools.
  - Single-call requests were honored.
  - No degenerate runs.
  - Over-length inputs fail the way the harness profile requires.
- **Measure first-event latency against the harness cutoff.** Long prefills need early-open
  plus keepalive.

## 5. Shadow traffic (live)

This needs a serving gateway that can mirror live requests to a target outside its serving
pools.

- **Setup:** point a mirror at an instance of the new model or build, with request tracing on.
- **What it gives:** a validation window of real, current harness traffic, and the client never
  sees the mirror's answers.
- **Lifetime:** a gateway may clear a mirror when its target fails; check it and re-arm it.
- **Keys:** keep `prompt_cache_key` (or an explicit session id) in replayed and mirrored
  requests. Without one, a gateway that keys a session by a hash of its first messages (e.g. the
  first two) puts every bare `ping` under one key.

## 6. Traps

- **Off-policy history is not a bug.**
  - A new model replying to another model's history may behave unlike it would on its own
    trajectory.
  - Count a model-class finding as the model's only after checking its raw text.
  - Never "fix" serving to hide model errors.
- **Caching and routing.** Replays of one thread share prefixes. Keep the session key in the body
  so a gateway routes them together, or caching numbers mislead.
- **Requests in other hours.** Requests and responses may land in different hour shards. Pass
  the boundary hour's request shards, or joins show up as `request_not_found`.
- **Corpora are sensitive.** Store them with the traces' permissions, and never commit one.
- **Slow first pass.** Parsing multi-megabyte request bodies dominates the cost; use `--workers`.
