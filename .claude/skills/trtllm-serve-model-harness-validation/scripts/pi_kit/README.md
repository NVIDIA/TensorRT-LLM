<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# pi kit: sandboxed pi sessions against a trtllm-serve endpoint

The kit runs real pi sessions on local tasks for an S4 run (`references/smoke.md` section 2) when no
hosted agent platform runs pi:
- each session runs in its own throwaway container;
- each session has its own `record_requests.py` in front of the endpoint;
- the trace is judged with this skill's validator like any other run.

`scripts/run_pi_sessions.sh` is the light alternative. It runs pi directly on the host, for wiring
checks against the fake servers (section 1 of `smoke.md`). The kit is for runs whose agents execute
model-generated commands.

## What is here

| Path | What |
|---|---|
| `bin/pikit` (`bin/pikit.py`) | The runner: `run`, `resume`, `stop`, `status` (`--help`). Python standard library only |
| `bin/session-entry.sh`, `bin/inbridge.mjs` | What runs inside each container |
| `presets/` | pi's model entry per served model (`compat` and `thinkingLevelMap`), so pi's thinking level reaches the model's own effort setting (`--compat-preset`; `contracts/pi.md` section 12) |
| `tasks/` | `tasks.tsv` and one probe task. Bring the cell's own tasks (below) |
| `selftest/fake_server.sh` | Starts and stops one of the skill's fake servers, recording the PID |
| `measure/` | Re-measures how a pi version handles tool failures, errors, cuts and silences against the fake servers: `cases.tsv`, `run.sh`, `probe.sh`, `summarize.py`, `source_checks/classify.mjs`. The pi 1.4.0 results are in `contracts/pi.md` |

## Requirements

On a docker host:

| Variable (or option) | What |
|---|---|
| `PIKIT_IMAGE` (`--image`) | The container image: whatever the tasks need (a shell, Python, compilers). It runs without a GPU and without network |
| `PIKIT_NODE_DIR` (`--node-dir`) | A Node.js install, mounted read-only |
| `PIKIT_PI_DIR` (`--pi-dir`) | A pi install (`npm install --prefix <dir> @earendil-works/pi-coding-agent@<version>`), mounted read-only |
| `PIKIT_PYTHON` (`--recorder-python`) | The validator's environment (`scripts/validate/requirements.txt`), for the recorders. Default: the Python running the kit |

`PIKIT_WORK` (fake servers) and `PIKIT_MEASURE_DIR` (measurements) set where the self-test and the
measurements write their output. By default they write under the current directory, never into
the skill.

## Tasks

`tasks.tsv` has one line per task: `id<TAB>kind<TAB>size<TAB>dir`. `<dir>` holds `prompt.md` (the
task as pi gets it) and `workspace/` (copied fresh for each session).

Nothing grades the tasks. They are there for realistic multi-turn tool use: read, write, edit,
bash, failing commands, long outputs. Use tasks that force several calls, a failing tool, long
outputs and, with a small `--context-window`, compaction (`smoke.md` section 2).

Keep tasks free of credentials and of code that may not leave its source. Tasks built from private
problem sets stay outside the repository; pass them with `--tasks`.

## The sandbox

Each session is one task in its own container (`pikit-<run>-sNNNN-aN`, labelled `pikit.run`,
`pikit.session` and `pikit.attempt`):

- **No GPU.** `--runtime runc` and `NVIDIA_VISIBLE_DEVICES=void`. A host whose default runtime is
  nvidia would otherwise expose its GPUs.
- **No network.** `--network none`.
  - pi talks to `127.0.0.1:18080` inside the container.
  - `inbridge.mjs` relays each connection to a unix socket in the session's `io/` directory.
  - The supervisor relays that socket to the session's recorder, which forwards to the endpoint.

  The endpoint is the only thing a session can reach. A cut on either side cuts the other.
- **No home directory and no credentials.**
  - Mounts: the session's copy of the task workspace (`/workspace`), pi's agent directory with a
    dummy key (`/pi`), the session's `io/`, and node, pi and `bin/` read-only.
  - `HOME` is a directory inside the container, and the image's `/home` is hidden.
  - The recorder forwards no credential headers.
- **Limits.** `--cap-drop ALL`, `no-new-privileges` and the caller's uid:gid. 2 CPUs, 8 GB and
  4096 pids by default (`--cpus`, `--memory`, `--pids-limit`).
- **pi's settings.** `-p --mode json --tools read,bash,edit,write`, offline. Extensions, MCP,
  skills, prompt templates, context files and themes are off. Everything else, timeouts, retries and
  compaction included, keeps pi's defaults: the settings the harness profiles measured.

A session's directory, `<run>/sessions/sNNNN/`, holds:

| Path | What |
|---|---|
| `state.json` | Status, attempt, container id, recorder pid, pi's exit code, a summary of its events |
| `io/events.jsonl`, `io/stderr.log`, `io/entry.log` | pi's JSON events, its stderr, the container entry's log |
| `pi/sessions/*.jsonl` | pi's own session log |
| `trace/` | The recorder's two files, session tag `<run>/sNNNN` |
| `workspace/` | What the agent left |
| `attempts/aN/` | An attempt that did not finish |

`<run>/sessions.tsv` has one line per session. `pi -p` exits 0 even when the run ends on an error,
so read `stop_reason` (pi's last `stopReason`) there.

The statuses:

| Status | Meaning | Final |
|---|---|---|
| `done` | pi exited by itself | yes |
| `timeout` | Past `--timeout-s` | yes |
| `stopped` | Stopped by `pikit stop` | no |
| `error` | The kit could not start the session | no |

- `resume` re-runs the sessions that are not final; `--rerun timeout` adds those.
- A second `run` on the same run directory resumes it if the settings are the same, and refuses
  otherwise.
- `stop` signals the supervisor's recorded pid, which stops the containers it started (by id) and
  their recorders. With no supervisor alive, `stop` stops the recorded container ids and recorder
  pids itself. Nothing is killed by name pattern.

## Running a cell

1. **Serve the model** from the build under test (`smoke.md` section 3d), with the model profile's
   `context.serving` configuration and the raw-output hook. Note:
   - the served model name;
   - `max_seq_len`;
   - the build id (the `pip show tensorrt_llm` version, which carries the commit).
2. **Write the run's `matrix.yaml`** from `templates/matrix.yaml`, outside the repository.
3. **Start the run.** It is detached and survives the shell:

   ```bash
   KIT=<skill>/scripts/pi_kit
   $KIT/bin/pikit run --run-dir <run> --tasks <tasks.tsv> \
     --api openai-completions --base-url http://<host>:<port>/v1 --model <served name> \
     --context-window <served max_seq_len> --max-tokens <the model's own agentic output cap> \
     --build-id '<version+commit>' --compat-preset <preset> [--thinking <level>] \
     --concurrency 8 --timeout-s 3600
   $KIT/bin/pikit status --run-dir <run>     # any time
   $KIT/bin/pikit stop --run-dir <run>       # stops only this run's sessions
   $KIT/bin/pikit resume --run-dir <run>     # continues what did not finish
   ```

   - `--sessions N` runs N sessions over the tasks in turn (default: one per task).
   - `--only s0003,s0007` runs some of them.
   - Every session gets its own recorder, so the validator knows the threads: Chat and Messages
     carry no session identity of their own.
   - `status` shows the run's preset and thinking level. Record both with the cell's verdict.
4. **Judge it** (fast tier) in the validator's environment, over every session's trace:

   ```bash
   python <skill>/scripts/validate/verdict.py \
     --harness <skill>/profiles/harness/pi-chat.yaml --model <skill>/profiles/model/<id>.yaml \
     --matrix <run>/matrix.yaml --population smoke --build-id '<version+commit>' \
     --raw <raw-output dirs> --out-dir <out> <run>/sessions/s*/trace
   ```

   Read `verdict.json`, not only the exit code. At a stage exit, freeze the run directory with a
   `MANIFEST.sha256` and hand it to the validator agent for the full tier (`SKILL.md` sections 4-5).
   The traces hold the agents' code and tool output, so never print their content.

## Protocol, output cap and effort per model

- **Use `openai-completions` (profile `pi-chat`)**, the API pi's guide configures for vLLM, SGLang and
  other self-hosted servers. `pi-responses` and `pi-messages` runs of the same tasks make a protocol
  differential (`smoke.md` section 3a).
- **Set `--context-window` to the served `max_seq_len`.** pi compacts on its own 16,384 tokens below
  it. On main, a request past `max_seq_len` is refused with OpenAI's over-length wording, which pi
  reads: after tool turns it compacts and sends the turn again, once per run. The session ends
  instead on an over-length first request (nothing to compact), and on refusals pi does not read:
  an older build's wording, or a prompt past `max_num_tokens` with chunked prefill off
  (`contracts/pi.md` section 11).
- **Set `--max-tokens` to the model's own agentic output setting.** For example, some GLM-5.3
  replies at effort Max ran 44k to 98k tokens, and a 32,768 cap would cut them at `length`.
- **Use `--compat-preset`** to send pi's thinking level where the model reads it, and to map pi's
  seven levels onto the model's own (`contracts/pi.md` section 12). `--compat` adds build-dependent
  flags on top, key by key. For example, `{"supportsDeveloperRole": false}` on a build whose chat
  path drops `developer` messages for the model's template.
- **Without a preset,** pi sends its level as a top-level `reasoning_effort`. Over Chat, only a model
  extension that maps it (Kimi-K3's) reads it there. `--thinking minimal` is then rejected with a
  400.

## Self-test, without a model

```bash
export PIKIT_IMAGE=... PIKIT_NODE_DIR=... PIKIT_PI_DIR=... PIKIT_PYTHON=<validator venv>/bin/python
$KIT/selftest/fake_server.sh start ok agent 18399 ok --max-tool-turns 1
$KIT/bin/pikit run --run-dir st --api openai-completions --base-url http://127.0.0.1:18399/v1 \
  --model fake-model --context-window 32768 --max-tokens 4096 --build-id selftest \
  --concurrency 1 --timeout-s 300 --foreground
$KIT/bin/pikit status --run-dir st
$KIT/selftest/fake_server.sh stop ok
```

The probe session ends `done` after two requests and one tool call, and no container is left.

- **Expected fast verdict over its trace** (`pi-chat`, any model profile):
  - inventory, conformance and runtime: PASS;
  - model-behavior and harness-outcome: BLOCKED, for too few units;
  - `H-01:tool_error` flagged, because the fake server's call reads a file that does not exist.
- **What it proves:** the kit's evidence is complete. It proves nothing about a model.
