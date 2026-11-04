<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# S4: smoke-test the real harness

Replay checks single turns. It cannot show what the harness does with our replies.

A serving bug can be invisible on the server side. For example:
- every call was well-formed, but a custom tool went out as a `function_call`;
- Codex answered `aborted` to those tool results;
- agents concluded their tools were broken.

A smoke run runs the harness itself, then reads the harness's verdict back from its own next
requests.

For a harness with no recorded traffic yet, S4's wiring run is also how its first corpus is made.

## 0. Isolation and safety

- **Agents run model-generated commands.** Use a disposable container or VM per run, with no
  access to the user's home, shared workspaces or cluster credentials.
- **One configuration home per session,** never the user's:

  | Harness | Variable |
  |---|---|
  | Codex | `CODEX_HOME` |
  | Claude Code | `CLAUDE_CONFIG_DIR` |
  | pi | a fresh `HOME`, or pi's own directory variable |

  A dev host holds harness credentials (e.g. `~/.codex/auth.json`, Claude MCP credentials). None
  of it may be visible to a session.
- **No real credentials in the environment the agent sees.** Agents print their environment and
  headers (e.g. with `curl -v`), so any credential in reach ends up in the trace.
- **Restrict egress.** A harness's own calls to its vendor fail under no egress, and that is
  expected noise. E.g. Claude Code's fast-mode check and WebFetch domain check call
  `api.anthropic.com` directly.

## 1. Wire it up without a model first (minutes, no GPU)

The same four parts for every harness: a fake server of the harness's protocol, the recorder in
forward mode in front of it, the harness's session runner, and a verdict over the recorded trace.
Codex over Responses:

```bash
python scripts/validate/fake_responses_server.py --port 8399 --max-tool-turns 2 &
python scripts/record_requests.py --out-dir run/trace --port 8397 --forward http://127.0.0.1:8399 &
printf '%s\t%s\n' ./task_src "Run hello.py and tell me what it prints." > tasks.tsv
BASE_URL=http://127.0.0.1:8397/v1 MODEL=fake scripts/run_codex_sessions.sh tasks.tsv run/out
python scripts/validate/verdict.py --harness profiles/harness/codex-default.yaml \
    --model profiles/model/<id>.yaml --population smoke --out-dir run/verdict run/trace
```

How it behaves:
- The fake server answers with a call to the first declared tool, and with `--max-tool-turns`
  it ends the turn.
- Codex executes the call and finishes.

Verified with codex-cli 0.155.1: every session exited 0, and every stream and call was
conformant.

**Claude Code.** Same wiring, with `fake_agent_server.py` and `run_claude_sessions.sh`:

```bash
python scripts/validate/fake_agent_server.py --port 8400 --max-tool-turns 1 &
python scripts/record_requests.py --out-dir run/trace --port 8397 --forward http://127.0.0.1:8400 &
printf '%s\t%s\n' - "Read the file named x and summarize it." > tasks.tsv
BASE_URL=http://127.0.0.1:8397 MODEL=fake-model scripts/run_claude_sessions.sh tasks.tsv run/out
python scripts/validate/verdict.py --harness profiles/harness/claude-code.yaml \
    --model profiles/model/<id>.yaml --population smoke --out-dir run/verdict run/trace
```

- **Run every session under `env -i`.** A parent agent session exports variables a child must
  not inherit. E.g. Claude Code exports `CLAUDE_CODE_*` variables, including a messaging socket
  and token. The runner does this.
- **Verified with Claude Code 2.1.289:** every session exited 0, every request and reply parsed,
  echo equal. The measured facts are in `contracts/claude-code.md` section 9.
- **`fake_agent_server.py --mode <fault>`** shows what the harness does with a broken reply, e.g.
  `undeclared_tool` and `invalid_input` gave the measured tool-error strings.

**pi.** `scripts/run_pi_sessions.sh` runs `pi -p --mode json` per task under `env -i`, with its
own `HOME` and `PI_CODING_AGENT_DIR` (a generated `models.json`), and startup network activity,
version checks, telemetry, extensions, MCP, skills and context files off. `API` picks the wire.

```bash
python scripts/record_requests.py --out-dir run/trace --port 8397 --forward http://127.0.0.1:8400 \
    --session pi-s1 &                    # pi sends no session identity over Chat or Messages
API=openai-completions MODEL=fake-model BASE_URL=http://127.0.0.1:8397/v1 TOOLS=read \
    scripts/run_pi_sessions.sh tasks.tsv run/out
```

- **Verified with @earendil-works/pi-coding-agent 1.4.0** over all three APIs against the fake
  servers: every session exited 0 and every request and reply parsed; text and reasoning came
  back verbatim (`contracts/pi.md` section 9).
- **For anthropic-messages, the runner strips a trailing `/v1` from the base URL:** pi appends
  `/v1/messages` for that API only (`contracts/pi.md` section 9).

**A wiring failure is a harness configuration problem, not a model problem.** Fix it before
spending GPU time.

## 2. The real run

1. **Serve the model with tracing on:**

   ```bash
   TRTLLM_REQUEST_TRACE_DIR=run/trace trtllm-serve <model> --tool_parser <name> \
       --reasoning_parser <name> [--custom_tokenizer <name>] ...
   ```

   - The server's request trace is not on main yet (`contract.md` section 1). Without it, put
     `scripts/record_requests.py` in forward mode in front of the server, as in section 1.
   - Put a gateway in front of it if the run should look like production. Keep session affinity,
     and serve the model name the harness sends.
   - Configure the harness's context window to the served `max_seq_len`:
     - Codex: `model_context_window`;
     - Claude Code: its model config;
     - pi: `contextWindow`.

     Otherwise compaction triggers at the wrong point.
   - Make the harness's thinking level reach the model's own effort setting, and record the level
     it ran at. E.g. pi over Chat sends its level as a top-level `reasoning_effort` by default,
     which only Kimi-K3 reads: give its model entry the `compat` and `thinkingLevelMap` of
     `contracts/pi.md` section 12 (`COMPAT` and `THINKING_LEVEL_MAP` for `run_pi_sessions.sh`,
     `--compat-preset` in `scripts/pi_kit`).
   - Serve the production serving configuration (section 3c): full context, chunked prefill
     (`enable_chunked_prefill: true`) with 8192-token chunks (`max_num_tokens: 8192`), block
     reuse on (off only for a protocol differential, section 3a). Keep it while it works: a smoke
     run is not where serving gets tuned.
   - Three settings bound three different lengths:

     | Setting | Bounds |
     |---|---|
     | `max_seq_len` | One request's whole context: prompt plus output |
     | `max_num_tokens` | One engine iteration, shared by every request in the batch (prefill and decode tokens). With chunked prefill on, it is the prefill chunk |
     | `max_tokens`, per request | The output. E.g. Claude Code sent 32,000 on every request |

   - Without chunked prefill, every prompt must fit in `max_num_tokens`. E.g. Claude Code 2.1.289
     opens with about 15.5k tokens, so with `max_num_tokens: 8192` DeepSeek-V4.1-Flash answered
     every Claude Code request with HTTP 400. Turn chunked prefill on rather than raise
     `max_num_tokens` to fit the prompt: a larger chunk can break long context (section 3c:
     DeepSeek-V4.1 at 65,536).
2. **Choose tasks the agent can complete locally.** Small repositories with scripted bugs work.
   Include tasks that force:
   - several tool calls;
   - a failing tool;
   - long outputs;
   - a long session that reaches compaction (set the window low to trigger it);
   - a sub-agent, where the harness has them.
3. **Run enough sessions to see rates, not anecdotes.**
   - Tens of sessions for wiring; hundreds before a large run (S5).
   - Set concurrency to reflect real load, since prefill under load is what trips harness
     timeouts.

## 3. Judge both sides

**Both sides at once.** `verdict.py --population smoke` judges the server side (`conformance`,
`runtime`) and the harness side (`harness-outcome`) from the same trace. Add
`scan_credentials.py` when tasks touch secrets.

**Harness side.** Each call's first reported outcome is read from the next request of its
thread. The strings come from the harness profile's `outcomes`, measured for its harness version
(`contract.md`).

| Outcome | Meaning |
|---|---|
| `aborted` | The harness cancelled the call. A call in the wrong shape is the usual cause, e.g. a custom tool delivered as a function |
| `unsupported` | The harness does not know the tool |
| `arguments_rejected` | The arguments do not match the tool schema |
| `unavailable_in_mode` | The tool is not available in the harness's current mode |
| `result` | Anything else: the call ran (success is not implied) |

Also counted:
- identical resends, which mean a reply the harness did not understand (e.g. a 400 where it
  expected an in-stream failure);
- client disconnects, often a first event later than the harness's idle cutoff;
- failed terminal codes;
- compaction requests, and whether they were answered substantively.

### Exit criteria

Set these in the run's matrix file with the human before starting; `validate.md` section 3 has
the rate rules.

- **Zero** serving-shape and contract failures, and views equal on 100% of replies.
- **`aborted` and `arguments_rejected` below an agreed rate,** each case traced to the raw
  generation.
- **Zero identical resends** caused by our error mapping.
- **Every disconnect attributed** to the client, a timeout or the server.

## 3a. A protocol differential

One harness, one task, one served model, several wires. A harness that speaks several protocols
can run the same task over each: pi speaks Chat Completions, Responses and Anthropic Messages, so
it can run the same task three times. `scripts/validate/protocol_diff.py` compares the runs turn
by turn in one protocol-free form: what the harness sent, the input the server billed, and what
the server delivered.

- **Make the requests equal**, or nothing on the server's side can be attributed.
  - Pin what the harness writes differently per run or per wire. For pi:
    - one fixed workspace path for every run (`WORKSPACE`): pi writes its working directory into
      the system prompt;
    - the same tools: pi adds `"required": []` to schemas without required parameters over
      Messages only, so allow tools that have them (`TOOLS=read`);
    - no reasoning parameters: pi writes "thinking off" three different ways (nothing,
      `reasoning.effort: none`, `thinking.type: disabled`), so declare the model non-reasoning
      (`REASONING=false`) and let each protocol path apply its default.
  - Greedy decoding at the recorder (`record_requests.py --override temperature=0`). The
    recorded body stays what the harness sent.
  - Deterministic serving: one session at a time and no KV-cache block reuse
    (`kv_cache_config: {enable_block_reuse: false}` in `trtllm-serve --config`). With reuse on,
    token-identical prompts gave different greedy generations.
    A model that is not deterministic even then (e.g. Qwen3-30B-A3B, an MoE) leaves the S4
    differential inconclusive after its first divergence: take render-level evidence from an S3
    replay instead, which gives every path the same requests.
- **Read the attribution, per turn:**

  | Attribution | Meaning |
  |---|---|
  | `equal` | Same request, same billed input, same reply |
  | `harness` | The harness sent something else; nothing on the server's side is judged |
  | `harness_literal` | The same meaning written differently (an empty `required`); not attributable |
  | `server_render` | Same request, replayed reasoning included, other billed input: the protocol path rendered another prompt |
  | `server_output` | Same request and billed input, the same generation delivered differently (another channel split, edge whitespace, other calls): the protocol path converted the output differently |
  | `server_rejected` | One protocol path refused or broke off a request another path served |
  | `divergent_generation` | Same request and billed input, text that differs mid-way: two generations, not a conversion. Serve deterministically to settle it |
  | `after_divergence` | The history carries an earlier difference, a reasoning-only one included: replayed reasoning that differs makes the bill differ too |

- **Exit codes:** 0 no server-side difference, 1 a server-side difference, 2 not conclusive.
- **A `server_render` is settled offline.** Render both requests through the served build's own
  prompt paths and compare the token ids: the shared render pipeline (`serve/render`,
  `render_chat(..., tokenize=True)`, which `count_tokens` also uses since #19964),
  `responses_utils._create_input_tokens`, and `anthropic_adapter.convert_anthropic_request`. On
  builds before #20775, `chat_tokenization.render_chat_request_for_tokenizer` is only the
  simplified fallback (media, `TRTLLM_RENDER_LEGACY=1`). The first differing token names the
  cause.
- Judge each run on its own too (`verdict.py`, with each wire's harness profile, e.g.
  `profiles/harness/pi-{chat,responses,messages}.yaml`).

## 3b. Fix a serving defect and prove it (A/B)

**Look for an existing fix first.**
- Open upstream PRs on the same code: list them (`gh pr list --state open --limit 1500`) and
  grep the titles. A port must not collide with one; extend that PR or stack on it.
- Downstream or deployment branches your organisation carries may hold harness fixes found under
  production traffic. Search them by code path and symbol:
  `git log -S<symbol> origin/main..<branch> -- <path>`.
- A branch commit seldom applies to main as is, because main receives this work as reworked
  squash PRs. `git merge-tree --write-tree --merge-base=C^ origin/main C` shows the conflicts
  without a worktree. Port the small ones, rewrite the large ones on main's structure, and keep
  their regression tests.

**Then prove the fix on one node.**
- A = main; B = main plus the fix, or plus every pending fix when their rules are disjoint.
  Python-only fixes share one C++ build: copy A's ignored build artifacts (`git status --ignored`
  under `tensorrt_llm/` and `3rdparty/`) into B's checkout.
- Serve both side by side on separate GPUs, with the same model and the same deterministic
  configuration as in section 3a.
- Give both the same inputs: one frozen corpus replayed (S3) and the same harness tasks (S4).
- Judge each side with `verdict.py`. The fix's rule goes to zero on B and every other count stays
  where it was. `protocol_diff.py` per side shows render-level fixes.
- Each regression test must fail on A's code by assertion. An import of a helper the fix adds
  fails at collection, which proves nothing: test behavior through the server instead, building
  it with its own constructor and patching out only model loading and the route table.

**Then hand it over.**
- Opening the PR is the human's call (`SKILL.md` section 4). Write its body from the A/B in public
  terms: the model, the harness versions, the build and the counts. No internal system or cluster
  names.
- Once it is open, track the PR's CI and reviews until it merges, merge conflicts included.
- Before reporting a PR's state, read it from GitHub, not from memory.

## 3c. When the output goes wrong: long context and the raw output

An agent that reads its task correctly and then loses it, mid-session, is usually not the model.
The symptoms: invented paths, a different task than the one it was given, a one-word final answer
("Received"), an early end_turn with nothing submitted. Check the serving configuration and read
what the model wrote before blaming the model.

**Serve the production serving configuration:** full context, chunked prefill with 8192-token
chunks (`enable_chunked_prefill: true`, `max_num_tokens: 8192`), block reuse on.
- Serve a new model the same way, at its full context, and keep that configuration while it
  works. Each model profile records its own under `context.serving`.
- Whatever the model sizes by position must cover `max_seq_len` too. E.g. Kimi-K3's MLA RoPE
  table is indexed by absolute position, so `KIMI_K3_MLA_MAX_POSITIONS` must cover `max_seq_len`.

**The prefill chunk, not the context length, can bound long-context recall.** E.g. DeepSeek-V4.1,
chunked prefill on:

| `max_seq_len` | `max_num_tokens` | Prompts longer than 65,536 tokens |
|---|---|---|
| 262,144 | 65,536 or 131,072 | Everything before the last 65,536 tokens lost |
| 1,048,576 | 65,536 | Lost the same way |
| 262,144 | 8192 | Recall held (probed to 130k) |
| 1,048,576 | 8192 | Recall held (probed to 417k) |

- The loss happened even when only 2k new tokens were prefilled on a cached prefix. Agents
  derailed on their first request past 65,536 tokens.
- Under the production serving configuration, agent runs went far past 65,536 tokens and kept
  their tasks.
- **Change one setting at a time.** Adopting the production serving configuration changed
  `max_seq_len` and `max_num_tokens` together, so the larger context looked like the fix until a
  bisect separated them.
- **A size that matches a failure is a suspect, not a cause.** V4.1's sparse indexer searches a
  candidate pool of 2048 blocks of 8 compressed positions, 65,536 tokens at its compress ratio of
  4, and only a subset of the blocks past that. The pool matched the loss exactly, yet with
  8192-token chunks recall held far beyond it.
- Chunks between 8192 and 65,536 are untested, and the cause in the code is not located. A
  defect like this is not the cell's to fix (`SKILL.md` section 4): write it up, hand it to the
  human, and go on testing the protocol under the configuration that avoids it.

**Probe recall before any agent run.**
- A clean recall probe: a unique passphrase inside natural filler, at 2% and at 50% depth. Probe
  sizes on both sides of every size the model or the configuration fixes (a chunk, a candidate
  pool, a RoPE table), up to the served maximum. Filler that repeats the passphrase's words tests
  distractors, not recall.
- Replay the first request an agent derailed on, at temperature 0, against each candidate build
  or configuration. A coherent answer there is the end-to-end check of a fix.
- Read the trace by agent run: the prompt size at each request (`input_tokens` plus
  `cache_read_input_tokens`) against those sizes. Several agents derailing at the same size is a
  serving bound, not several model failures.

**Read what the model wrote: the raw output.**
- The raw-output hook (`tensorrt_llm/executor/raw_output_hook.py`) records every generation before
  any parser touches it. A trial build gets it from its `raw-hook` item (section 3d); main already
  has `--post_processor_hook`. Serve with `TRTLLM_RAW_OUTPUT_DIR=<dir>` and
  `--post_processor_hook tensorrt_llm.executor.raw_output_hook.RawOutputDump`.
- One JSONL per UTC hour, one record per (request, output): the text the model generated, the
  streamed frames kept apart, start and finish times, and the engine request id, which is the
  `chatcmpl-<id>` of the reply.
- Hold it beside the trace:
  - the raw text is already wrong: the generation is at fault (configuration, attention, kernels);
  - the raw text is right and the reply is not: a parser or the endpoint adapter (C-13 names it);
  - a marker split across two frames: an incremental-parser seam.
- `verdict.py --raw <dir>` runs C-13 and USG-06 over the same records. Whenever the served build
  recorded raw output, pass `--raw`: without it C-13 reads NOT_APPLICABLE and can turn a BLOCKED
  claim into a PASS. NOT_APPLICABLE is right only for a build without the hook.

## 3d. Compose the build under test

A cell runs on a trial build: a base, usually the current main, plus every open PR the cell needs,
plus the capture changes. `scripts/compose_trial_build.sh` builds it:
- each item lands as one commit: its diff against its merge-base with main, applied three-way;
- what went in is appended to a provenance file.

```bash
scripts/compose_trial_build.sh -p <run>/provenance.txt <worktree> <branch> origin/main \
    raw-hook=<ref> engine-id=<ref> disagg-id=<ref> <label>=pr:<number> ...
```

Each `<ref>` is a `pr:<number>` or any commit-ish (a branch, a commit) that carries the change.

**The capture changes make the evidence joinable.** Until they are in main, add each one from
wherever it is carried:

| Item | What it adds | Without it |
|---|---|---|
| `raw-hook` | The raw-output hook (section 3c) | No raw view: C-13 and M-02 cannot be judged |
| `engine-id` | Responses replies name their engine request: `/v1/responses` replies carry its id in `x-trtllm-engine-request-id`, which the recorder stores | A Responses reply joins no raw record: its id is a random `resp_` id |
| `disagg-id` | Disaggregated `/v1/responses` replies are named `resp_<disagg_request_id>_<hex>`, which the validator reads first | Generation workers number their requests independently, so an engine id names a request on each worker |

On main, a streamed Chat Completions reply is already named `chatcmpl-<engine id>`. A non-streamed
one is named `chatcmpl-<random>`, and a Messages reply `msg_<random>`. PR #20936 names both after
the engine request; until it merges, add it to a build that serves either.

**Order and conflicts.**
- Apply the capture changes first, then the PRs in the order they stack.
- A conflict stops the script. Resolve it, commit with the printed message, and rerun with `-k` and
  the remaining items.
- Two PRs that change the same test expectation pass apart and fail together. Record the clash in
  the cell log and on both PRs. It is not a serving defect.

**Install, test, name.**
- **Install.** Python-only items share the base's C++. Install with
  `TRTLLM_PRECOMPILED_LOCATION=<a wheel whose C++ matches the base>` and `pip install -e .`. An item
  that changes C++ needs a full build. Each concurrent job gets its own checkout (section 5).
- **Test.** Run on the trial build, CPU only:
  - every item's own tests;
  - the serve, tool-parser, reasoning-parser and executor suites.

  A failure the base shares is not the items'.
- **Name.** The installed version (`pip show tensorrt_llm`: `<version>+<commit>`) is the recorder's
  `--build-id` and the verdict's. A rebuilt branch is a new build: a new id and a new trial.

**Keep the provenance file with the evidence.**
- It names each PR head the cell ran. A PR that moves after the trial is not proven by it.
- A PASS on this build is a PASS on the trial build, not on main (`SKILL.md` section 3a).

## 4. What a smoke run still cannot show

Long-tail rates. Some defects show once in 10^5 replies or less, or only under load:
- a parser defect, e.g. two calls delivered under `parallel_tool_calls=false` when the model left
  out a call's close tag;
- a kernel race that corrupts a decode step only under load, e.g. Kimi-K3's `[PAD]` runaways.

Both need 10^5–10^6 turns at production concurrency. That is S5: a large run (a soak, or a
production-like campaign on a platform that runs the harness), still validated with the same
checks.

## 4a. Validate a large run (S5)

A large run is judged like any other run, the same way every time. A run that was never validated
has no result to report. A fast verdict, least of all one without `--raw`, is never reported as a
PASS (`SKILL.md` section 5).

1. **Freeze it when it ends.** Copy the recorder's two files and the raw-output hour files into a
   snapshot directory with mode 0700: agents print their environment, so the traces can carry
   credentials (scan them with `scripts/validate/scan_credentials.py`). Then write
   `MANIFEST.sha256` over the copies. Never hash live files.
2. **Run the fast verdict on the snapshot,** with the cell's profiles. Pass `--raw` whenever the
   build recorded raw output, and `--build-id` with the build under test as deployed: the value
   the recorder was started with, never one copied back from the records, which are checked
   against it. Without it, inventory is BLOCKED:

   ```bash
   python scripts/validate/verdict.py --harness profiles/harness/claude-code.yaml \
       --model profiles/model/<id>.yaml --population smoke --raw <snap>/raw \
       --build-id <the recorder's --build-id> --out-dir <snap>/verdict <snap>/trace
   ```

3. **Hand the snapshot to the validator agent for the full tier,** in another session. It
   verifies the manifest and runs the control matrix for the exact validator tree. Then it runs
   `verdict.py --tier full --controls-receipt controls.json`, re-derives the key numbers from the
   raw bytes with its own scripts, and writes the report that `validate.md` section 5 describes.
   Only this verdict sets the cell's status.
4. **Report each claim with its tier and inputs** (`SKILL.md` section 5). Read every FAIL to its
   locator first, and say what it is: the server's reply, the recorder's own error reply, or the
   harness re-asking.

A platform's own per-agent logs explain single sessions. They are not validator input
(section 5).

## 5. Traps

- **Configuration changes the contract.** Smoke-test the configuration production uses, under
  its own harness profile (`contract.md` section 5).
- **A fake server that always calls a tool never ends the conversation.** E.g. Codex looped for
  120 s and sent 1.9 GB of history. Use `--max-tool-turns`.
- **Harness logs are a secondary signal.**
  - A harness's own log (Codex `--json`, Claude Code `stream-json`, pi's session JSONL) explains
    individual sessions.
  - The rates come from the server's trace, which records exactly what the harness sent back.
- **Harnesses rewrite what they send back.** E.g. Claude Code 2.1.289 adds an argument's schema
  default to an echoed call (Edit `replace_all: false`) and drops a leading `cd <cwd> &&` from a
  Bash command. Measure such a rewrite on the raw traces first (every instance, nothing else
  changed). Then declare it in the harness profile's `echo.tool_input`. Never declare one to make
  a FAIL go away.
- **Every concurrent job gets its own checkout.** `pip install -e` with
  `TRTLLM_PRECOMPILED_LOCATION` re-extracts the compiled libraries in place. Installing into a
  live server's checkout can kill that server: every rank, with SIGBUS.
- **Run the harness the way production does.** A wrapper (LiteLLM, a gateway) can add, drop or
  rename fields.
- **A reply with no first byte is not always prefill.** Serve with iteration stats on and sample
  `/metrics` every 2 s, so a request queued in the engine can be told from one that never reached
  it. Against the raw output's first-token times, a silent request is one of three things:
  - a reply lost between the server and the client, after the engine produced it;
  - a slow prefill;
  - a request the server accepted that never produced a token while later ones ran, with nothing
    logged on any rank. Report it with its engine id; its cause is not known.
