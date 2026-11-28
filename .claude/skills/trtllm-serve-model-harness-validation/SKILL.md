---
name: trtllm-serve-model-harness-validation
description: >-
  Support a model x agent-harness pair on trtllm-serve and prove it, as an
  agent loop: capture the harness's wire contract, adapt the model's
  rendering, parsers and stop rules, replay recorded traffic, smoke-test the
  real harness, and let an independent validator issue fail-closed verdicts
  per cell (inventory, conformance, runtime, model behavior, harness outcome)
  over Responses, Anthropic Messages and Chat Completions traffic. Works for
  any model and any agent harness: each is a profile, and the shipped
  profiles (GLM-5.x, Kimi-K3, DeepSeek-V4.1, Qwen3; Codex, Claude Code, pi)
  are examples. Triggers on: new model for agents, new harness, an agent
  harness against trtllm-serve, harness contract, tool parser for agents,
  reasoning parser, replay traces, smoke test a harness, validate a serving
  change, stream/final mismatch, parallel_tool_calls, model x harness
  matrix, agent support loop.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Support a model x harness pair on trtllm-serve

**Why.** When an agent harness runs on a self-hosted model, the expensive serving bugs are
usually harness-contract bugs, not model bugs:
- a custom tool delivered as a function, so the harness aborted its tool results;
- a field the harness always sends, rejected by the server;
- an HTTP 400 where the harness needed an in-stream failure, so it resent forever;
- minutes of silence during prefill, so the client cut the stream.

Each is cheap to fix once known. Each can be found without production scale, given the right
inputs and the right checks. That is what this skill provides:
- a procedure;
- profiles that describe each model and harness as data;
- a recorder;
- a validator that judges recorded traffic and fails closed.

## 1. The matrix

- **A cell is `(model profile, harness profile)`.** The harness profile fixes the protocol, so
  one harness over Chat Completions and over Anthropic Messages makes two cells.
- **Work per axis, verify per cell.**
  - Model adaptation (renderer, tool and reasoning parsers, stop rules) is shared by every
    harness.
  - trtllm-serve has two output paths:
    - `/v1/responses` has its own path;
    - `/v1/messages` is converted onto the `/v1/chat/completions` path, so the two share parsers.
  - Harness quirks live in the contract and the harness profile.
- **Profiles are data.** Both live under `profiles/`:
  - `profiles/model/*.yaml`: markers, special and degenerate tokens, parsers, renderer source,
    the serving configuration;
  - `profiles/harness/*.yaml`: protocol, routes, identity keys, outcome strings, timeouts,
    error mapping, the rewrites the harness applies to what it sends back.

  Both are derived from S0 and S1 outputs, and the validator reads them.

  A profile's `status` says how far it is proven. `harness-outcome` is judged only on a harness
  profile that is `measured` (its fields measured for the version it names); `partial` or `draft`
  leaves it BLOCKED. A model profile's `status` is for the reader (`draft` until the served
  checkpoint's renders read back, `native_goldens.py --check`); the validator does not read it.
  In any profile, a field still `TODO` leaves the checks that read it BLOCKED.
- **Adding a model or a harness is adding files.** No procedure step and no check names a
  specific model or harness; the shipped profiles and contracts are examples.

  | To add | Data | Code, only when needed |
  |---|---|---|
  | A model | `profiles/model/<id>.yaml` | The model writes an output format no reader in `scripts/validate/native.py` reads yet. Add a reader, selected by the profile's `native.kind`, and the checkpoint's golden renders to `scripts/validate/native_goldens.py` |
  | A harness | `contracts/<harness>.md` from `contracts/TEMPLATE.md`; one `profiles/harness/<id>.yaml` per protocol it speaks | A runner that launches it (`scripts/run_*_sessions.sh`, or a kit like `scripts/pi_kit/`). The harness rewrites what it sends back in a way no rewrite in `scripts/validate/turnlib.py` covers: add one, named in the profile's `echo.tool_input` |

- **State.** Copy `templates/matrix.yaml` to a run directory outside the repository. Each
  cell's status is computed from its latest verdicts, never typed by hand.

## 2. Roles

| Role | Does | Never does |
|---|---|---|
| adapter agent | Contracts, adaptation, fixes, replays, smoke runs, fix reports | Edits validator-owned files in a cycle, or writes a verdict |
| validator agent | Freezes evidence, runs and owns the checks and controls, issues full-tier verdicts | Runs the adapter's scripts as evidence, or fixes serving code |
| investigator (on demand) | Runtime/kernel root causes with env-gated diagnostics | Restarts serving |
| human | Resources, thresholds, external actions (pushes, PRs, issues), policy choices, entering S5 | — |

- **On your own, you are the human.** With one developer and one agent, the agent proposes and
  the developer decides. Before the first GPU run, decide and record in the run's matrix and the
  cell log, with who and when:
  - the scope and the checkpoints;
  - the thresholds (the template proposes 0.02 per turn);
  - the budget;
  - the S0 sign-off;
  - who runs the full tier.

  Until the thresholds are set, every rate reads BLOCKED, by design.
- **Any agent can run it.** The procedure needs a shell and Python, nothing of Claude Code's own.
  - Codex 0.155.1 finds skills in `~/.codex/skills` and in the repository's `.codex/skills`, not
    in `.claude/skills`. `.codex/skills/<this skill>/SKILL.md` points it here, with this file's
    `name` and `description` (the tests keep them equal).
  - Where the text names a Claude Code feature (`/loop`), use your agent's own way to wait for
    a job.
  - The validator agent can be a sub-agent, ideally of another vendor's model.
- **The validator should be a different session, ideally a different vendor's model.** It
  catches what the operator cannot see, e.g.:
  - a test count that came from a filtered subset;
  - evidence stored truncated;
  - a protocol violation that shows once in a large run.
- **Validator-owned files:** `scripts/validate/`, `references/validate.md`,
  `references/protocols.md`, `references/turn-ir.md`, `references/acceptance-cases.md`, and
  the thresholds in the run's matrix file. Changing any of them is a policy change:
  - a separate commit;
  - the control matrix re-run;
  - an iteration-log entry.

  It is never a fix to a failing cell.

## 3. Stages and exit gates

| Stage | Do | Exit gate | Read |
|---|---|---|---|
| S0 contract | Record the harness, read its source or docs, and write the contract and harness profile. Write the model format card and model profile | Human signs off once per harness version and configuration | `references/contract.md`, `contracts/` |
| S1 adapt | Golden renders against the model's own encoder; parser fault matrix; stop rules; regression tests | Renders byte-equal; zero `calls_differ`, `streams_disagree` and `text_lost`; every new test failed on the old code | `references/adapt.md`, `references/verify-logic.md` |
| S2 regress | L1 frame replay and L2 fake engine over existing corpora, for every model already supported | Fast-tier `conformance` PASS on every regression corpus | `references/replay.md` |
| S3 replay | L3: recorded harness requests, one turn at a time, against the new model | Full tier: `inventory` and `conformance` PASS; `model-behavior` within thresholds, or BLOCKED for insufficient n | `references/replay.md`, `references/validate.md` |
| S4 smoke | L4: the real harness on local tasks, in a sandbox the agent prepares (`references/smoke.md` section 0) | Full tier: S3's claims plus `runtime` and `harness-outcome` | `references/smoke.md` |
| S5 scale | A large run for long-tail rates: a soak, or a production-like campaign on a platform that runs the harness | Human decision. The same checks: frozen traces, a fast verdict, then the full tier | `references/smoke.md` section 4a |

A harness with no recorded traffic yet runs S4 wiring and recording before S3. Its first real
traffic comes from L4.

## 3a. The fast path: a pair whose layers already exist

Sometimes the model's renderer, tool parser and reasoning parser are already in main with their
tests, and the harness profile is measured. Then the question is whether the pair works end to
end, not how to write a layer, and a cell may go from S0 straight to real traffic:

1. **S0 audit.** Read main against the harness contract and the model's format card, read-only.
   List each candidate gap with its code path. A gap read from code is a candidate, not a
   defect (section 5).
2. **Confirm, then one PR per confirmed gap.** Replay what the harness was recorded sending, or
   what the model was recorded writing, through main's code path. A gap that fails there is
   confirmed: look for an existing fix first (section 4, step 4), then open a PR with a test that
   fails on the old code. Pure logic also gets a property with its mutation gate
   (`references/verify-logic.md`). Opening each PR is the human's call (section 4, step 7). A
   candidate nothing confirms goes into the cell log with what would show it (a task, a probe,
   a check); the real traffic of step 4 then decides.
3. **A trial build:** main, plus the PRs the cell needs, plus the capture changes
   (`references/smoke.md` section 3d). Run every item's unit tests on it.
4. **Real traffic.** S4 on local tasks, or an S5 run, judged like any run: the fast tier, then
   the full tier.

A new layer (a parser, a renderer, a stop rule) still climbs S1-S3. On the fast path, this is
what stands in for each skipped stage, and what nothing covers:

| Stage | Stands in | Not covered |
|---|---|---|
| S1 | The full tier's raw view of every reply: C-13 compares what was delivered with what the model wrote, read in its own format by a reader qualified on the served checkpoint encoder's renders (`scripts/validate/native_goldens.py --<model> <checkpoint> --check`, CPU only). USG-07: the billed input equals the server's own prompt count | Prompt fidelity: no check compares the served prompt render with the encoder's (S3's render-level A/B does). Shapes the traffic did not produce: the parser fault matrix, stop rules on rare tokens |
| S2 | Each PR's own tests and CI | The other models a shared layer serves (a parser base class, the chat route, a shared serializer). When a PR changes such a layer, run S2 for them |
| S3 | Nothing: the first real traffic is S4 or S5 | A render-level A/B on identical requests, one turn at a time |

- **Name what the run did not exercise** in the cell log. The cell does not prove a fix whose
  path the traffic never took, e.g. no request ran over length, or no call was abandoned.
- **A fast-path PASS is a PASS on its trial build.** Report it with the build's provenance file.
  It becomes a PASS on main only after a run on a main build that has every item merged. Until
  the capture changes are in main too, that run has no raw view (C-13 reads NOT_APPLICABLE),
  and the report says so. A model profile that names its degenerate tokens by id only (e.g.
  GLM-5.x) also leaves M-02, and so `model-behavior`, BLOCKED without the raw view: such a cell
  cannot PASS on a build without the raw-output hook.
- **Name the evidence by the stage the run was.** Local tasks are S4
  (`cells/<cell>/S4/<iteration>/`); a large run is S5.
- **Record the S0 sign-off** in the cell log: who signed, the harness version and configuration,
  and the profile's commit.

## 4. The loop (per cell)

1. **Pick work.** Choose the lowest-stage cell that is not PASS from the matrix status. Cells
   that share a model or path move together: one parser fix serves every harness.
2. **Run the stage** and write its evidence under the run directory: `cells/<cell>/<stage>/<iteration>/`.
3. **Fast verdict.** Run `scripts/validate/` on the evidence. Read the verdict JSON, not only
   the exit code (see section 7).
4. **On FAIL:**
   - look for an existing fix first, in open upstream PRs and in any downstream branch your
     deployment carries (`references/smoke.md` section 3b);
   - when the output itself goes wrong (an agent that loses its task mid-session), first serve
     the production serving configuration (full context, chunked prefill, 8192-token chunks),
     then read the raw output before blaming the model (`references/smoke.md` section 3c);
   - make the smallest reproducer;
   - write a test that fails on the old code. When the fix is in pure Python logic (a parser, a
     protocol path, request validation, a validator rule), also state the rule as a property
     over generated inputs and gate it with planted bugs (`references/verify-logic.md`);
   - fix the responsible layer (renderer, parser, protocol path or harness configuration);
   - prove it with an A/B on the same inputs (`references/smoke.md` section 3b), re-run S2 for
     every model the layer touches, then repeat the stage.

   A defect below those layers (attention, kernels, the scheduler) is not the cell's to fix.
   Avoid it with the production serving configuration where that works, write it up
   (reproducer, evidence, what a bisect has ruled out) and hand it to the human, who decides
   whether it is chased or reported. The cell goes on; if nothing avoids the defect, the cell is
   BLOCKED on it.
5. **At a stage exit:** freeze the evidence with a `MANIFEST.sha256` and hand it to the
   validator for the full tier. Update the status only from that verdict.
6. **Budget.** Allow at most 3 repair cycles per finding class. Stop when two consecutive cycles
   reproduce the same blocker with no new evidence. Adding evidence does not consume a cycle.
7. **Escalate to the human** for:
   - GPU or time beyond the matrix budget;
   - pushes, PRs or upstream issues. Track an open PR's CI and reviews until it merges;
   - a defect below the cell's layers (step 4);
   - a policy choice, e.g. what serving should do when the model breaks a constraint;
   - a threshold;
   - an exhausted budget.

## 4a. The outer loop: drive a cell until it passes

The loop above fixes what one stage's verdict finds. The outer loop keeps a cell moving through
deploy, test, real-harness and large runs, validation and fixes, until the cell passes. Drive it
with whatever your agent has for waiting on a job: a self-paced loop (Claude Code's `/loop`), or
a session that a background watcher on each job wakes. Either way, a deploy or a run wakes it
when its job ends, not a timer. Take one step per iteration.

- **State.** The run's `matrix.yaml`; the evidence under `cells/<cell>/<stage>/<iteration>/`; and
  `cells/<cell>/log.md`, one line per iteration: the step, its evidence and its verdict. A cell's
  status comes from its verdicts only.
- **Each iteration:**
  1. Read the cell's latest full-tier verdict and the fast verdicts since. List the blockers: the
     failing checks and every non-PASS claim's `claim_blockers`.
  2. Take one step for the first blocker the table below can act on. Deploy the model profile's
     `context.serving` configuration from the trial build (`references/smoke.md` section 3d),
     with the raw-output hook and the recorder's `--build-id`.
  3. Freeze the new evidence and run its fast verdict (section 5). At a stage exit, hand the
     snapshot to the validator agent for the full tier.
  4. Append the log line, then stop or schedule the next iteration.

| Blocker | Step | Decides |
|---|---|---|
| A check FAILs | Step 4 above: an existing fix first, a reproducer, a test that fails on the old code, a property with its gate, the fix, the A/B. Track the PR until it merges | the loop; a PR needs the human |
| Evidence is missing: requests without a response, no `build_id`, no raw output, ids that join nothing | Fix the capture (recorder options, the raw-output hook, the build) and run again | the loop, within the GPU budget |
| The harness profile is `partial` or has TODOs: timeouts, outcomes, error wordings | Measure them (`references/contract.md`), against the fake servers first | the loop |
| A rate is BLOCKED for insufficient n | More sessions: a larger smoke run, then an S5 run | the human (S5) |
| Thresholds are unset | Propose values from earlier cells and production; the human sets them | the human |
| A defect below the cell's layers | Write it up and park it (step 4) | the human |

- **Stop when:**
  - every claim the cell requires is PASS at the full tier, from the validator agent: the cell
    passes;
  - or the next step is the human's (the right column): ask, and wait;
  - or the budget is spent: 3 repair cycles for one finding class, or two iterations on the same
    blocker with no new evidence.
- **Never** type a status, skip the verdict of a run, or report a fast verdict as the cell's
  status.

## 5. Rules that are not negotiable

- **Fail closed.** Missing evidence is BLOCKED, never PASS and never NOT_APPLICABLE.
  NOT_APPLICABLE needs a reason based on a feature that is absent.
- **No result without a verdict.** Every run that leaves a trace gets a verdict before its
  outcome is reported: a smoke run, each side of an A/B, an S5 run.
  - A report names each claim's tier and inputs: the population, the profiles, and whether
    `--raw` was given.
  - A fast verdict is a pre-check. It never stands in for a stage exit and is never reported as
    a cell's status.
- **Never weaken a gate to pass,** and never rename an objective to close one.
- **Never print trace content.** Use locators `(file, line, trace_id)`, lengths and SHA-256.
  Traces can carry credentials and user code. Never commit a corpus.
- **Prove what served.** Use process-level identity: `build_id` on every record, plus
  `provenance.txt`. A moved checkout proves nothing.
- **A regression test must fail on the old code.** A collection error on the old code is not a
  failure.
- **Fix only what is shown to fail.** A fix needs evidence that the failure happens:
  - a check that FAILs or flags on recorded traffic, with its locator;
  - or a reproducer on the build under test that replays recorded input (a request the
    harness sent, an output the model wrote) through the real code path.

  A code reading, synthetic input nobody was seen to send or write, or a case that only
  *would* happen is a candidate: log it with what would show it, and let the next run decide.
  A PR's description states its evidence, never a hypothetical. This applies to fixes of code
  that exists; a new layer (S1) is still built against the fault matrix.
- **Do not hide model behavior in serving.** Attribute with the raw generation first, and keep
  a conversation on one model. A failure inside the model's generation, e.g. a degenerate token
  run from a numerical or kernel bug, is detected and reported (M-02, R-04, the raw output) and
  fixed at its source. Never stop, cut or rewrite it in serving: that hides the symptom that
  leads to the bug.

## 6. Map

| File | Read when |
|---|---|
| `references/contract.md` | S0: capturing a harness and writing its contract and profile |
| `contracts/*.md` | The shipped harness contracts (Codex in its default and hosted configurations, Claude Code, pi) and the template for a new one |
| `references/adapt.md` | S1: renderer, parsers, stop rules, fault matrix, fault catalog |
| `references/verify-logic.md` | A fix to pure Python logic: its rule as a property over generated inputs, gated by planted bugs |
| `references/replay.md` | S2/S3: L1 frames, L2 fake engine, L3 per-turn replay, shadow traffic |
| `references/smoke.md` | S4: isolated real-harness runs, the serving configuration, harness-side signals; fixing a serving defect and handing its PR over (3b); long context and the raw output (3c); composing the build under test (3d); validating an S5 run (4a) |
| `scripts/pi_kit/README.md` | Running pi on local tasks, one container per session (S4): an example of a harness kit |
| `references/validate.md` | Any verdict: claims, the gate-to-check catalogue, rate rules, tiers, outputs |
| `references/turn-ir.md` | Writing or reading a protocol adapter |
| `references/protocols.md` | Wire rules for Responses, Anthropic Messages and Chat Completions |
| `references/acceptance-cases.md` | Controls that qualify the validator itself |
| `profiles/`, `templates/matrix.yaml` | Model and harness profiles; the run's matrix, thresholds (with a starting proposal) and budget |
| `profiles/model/serving/` | Complete serving configs, with their serve commands, for the shipped models |
| `tasks/` | A small public task set for real harness runs, in every runner's format (`tasks/README.md`) |
| `templates/full-tier-report.md` | The full tier's steps and the five files it writes |
| `tests/test_validator.py` | Checking the validator after a change (`MH_RUN_CONTROLS=1` for the matrix) |
| `tests/test_validator_properties.py`, `tests/validator_properties.mutations` | The worked example of `references/verify-logic.md`: the validator's own properties and their gate |

## 7. Scripts

| Script | Stage | Scope |
|---|---|---|
| `scripts/validate/verdict.py` | all | **The validator.** Responses, Anthropic Messages and Chat Completions; claims, rate bounds, exit codes 0/1/2/3 |
| `scripts/validate/sdk_diff.py` | all | A second reading of every reply by the official `anthropic` and `openai` SDKs (check C-12); run by `verdict.py` |
| `scripts/validate/native.py`, `native_goldens.py` | all | The raw view (C-13): the model's own text read in its own format; goldens rendered by each checkpoint's encoder, and `--check` on the checkpoint you serve |
| `scripts/validate/usage.py` | all | Usage rules (C-14): wire fields, totals, `count_tokens`, the engine's and the server's own counts |
| `scripts/validate/requirements.txt` | all | The validator's environment, SDK versions pinned: `python3 -m venv <dir> && <dir>/bin/pip install -r requirements.txt` |
| `scripts/validate/controls.py` | qualification | Control matrix for all three protocols plus cross cases; writes `controls.json` bound to the validator tree |
| `scripts/validate/fake_agent_server.py`, `fake_responses_server.py` | controls, S0, S4 wiring | Model-free servers, correct or broken on purpose; measurement modes for a harness's timeouts, retries and error mapping (`references/acceptance-cases.md` section 3) |
| `scripts/validate/mutate_traces.py` | qualification | Seeded faults on recorded Responses replies (negative controls on real traffic) |
| `scripts/validate/scan_credentials.py` | all | Any shard |
| `scripts/validate/check_stream.py`, `check_tool_calls.py`, `harness_signals.py`, `run_controls.sh` | Step 3 | Responses-only reference implementation, kept for calibration |
| `scripts/mutate.py` | S1, qualification | The mutation gate of a property: plants each bug of a `.mutations` file and re-runs the property (`references/verify-logic.md`) |
| `scripts/record_requests.py` | S0, S4 | Any route; record-only or forward; per-chunk timing |
| `scripts/summarize_capture.py` | S0 | Mostly Responses |
| `scripts/fault_mutations.py`, `scripts/chunk_equivalence.py` | S1 | Any tool parser; card-driven |
| `scripts/build_replay_corpus.py`, `scripts/replay_requests.py` | S3 | All three routes (`--route`); thread keys and reasoning transforms per protocol |
| `scripts/count_tokens_corpus.py` | S4 | A `count_tokens` probe from recorded Messages requests, replayed into the same trace (USG-05) |
| `scripts/run_codex_sessions.sh`, `scripts/run_claude_sessions.sh`, `scripts/run_pi_sessions.sh` | S4 | Session runners for the shipped harnesses (`env -i`, an isolated configuration each). On the host: for wiring runs against the fake servers |
| `scripts/pi_kit/` | S4 | pi on local tasks: one throwaway container and one recorder per session, `run`/`resume`/`stop`/`status`, model presets for pi's thinking level, re-measuring a pi version (`scripts/pi_kit/README.md`) |
| `scripts/compose_trial_build.sh` | S4, S5 | The build under test: a base plus open PRs and the capture changes, one commit each, with a provenance file (`references/smoke.md` section 3d) |
| `scripts/probe_recall.py` | S4, S5 | Long-context recall at chosen sizes and depths, before any agent run (`references/smoke.md` section 3c) |
| `scripts/validate/protocol_diff.py` | S4 | One harness and task over several wires, compared turn by turn; attributes each difference to the harness or to the server's protocol path |

- **The environment.** Creating it needs the package index (PyPI or a mirror) once; the tests,
  the controls and the verdicts then run offline.
- **Tests.** `tests/` holds CPU-only tests of the skill's own tools. Run them from a scratch
  directory, so nothing is written into the checkout:
  `PYTHONDONTWRITEBYTECODE=1 <venv>/bin/python -m pytest -q -p no:cacheprovider --basetemp=<scratch> <skill>/tests`.
  The trial-build tests make throwaway git repositories under `--basetemp`; no test touches a
  checkout.
- **The controls receipt** that a full tier needs:
  `<venv>/bin/python scripts/validate/controls.py --out-dir <dir>` writes `<dir>/controls.json`,
  bound to the validator tree it ran on (`MH_RUN_CONTROLS=1` runs the same matrix inside pytest,
  but keeps no receipt).
- **The mutation gate** (`scripts/mutate.py`) writes each planted bug into the real file and
  restores it. Give it no short time limit (tens of minutes for the validator's 62 bugs), and
  check that every `find` still matches once after any run that did not print its `SUMMARY`.
  When another agent may work in the same checkout, run it on a copy of the skill.
- **Run the validator in its own environment** (`scripts/validate/requirements.txt`). The SDK
  versions are part of what a verdict means: without an SDK, C-12 is BLOCKED for its protocols,
  and a control receipt qualifies only the SDK versions it was produced with.
- **Read the verdict JSON, not only the exit code.** A BLOCKED claim names its blocker in
  `claim_blockers`. Typical blockers:
  - a check that is BLOCKED itself, named as `<check>: <reason>` (a rate with too few units to
    decide against its threshold);
  - unset thresholds;
  - a harness profile that is not yet `measured`;
  - missing timing or `build_id`, or replies that carry a `build_id` while no `--build-id` was
    declared to check it against.
