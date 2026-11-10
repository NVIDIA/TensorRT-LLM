<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# trtllm-modeling-v2

Build a TensorRT-LLM ModelingV2 target from a HuggingFace checkpoint: reuse or certify catalog
operators, assemble the model, and prove the requested features. Agents write code; Python owns
the order of the work, integration and the verdicts. Performance tuning is another tool's work, on
the delivered target.

## Terms

| Term | Meaning |
|---|---|
| Target | Model implementation for a checkpoint family and supported deployment configurations |
| Unit | Core implementation or one feature added to it |
| Member | One parallel configuration, such as `tp4`, `tep4` or `dep4` |
| Variant | A unit's target served another way (a KV dtype, chunked prefill, a draft depth) |
| Check | One gate (boot, accuracy, serve, disagg, acceptance) for one unit or variant at one member |
| Contract | Every check the run must pass, compiled from the spec |
| Task | One piece of work for one agent turn: the survey, a kernel entry, a unit, a variant, a fix |
| Entry | A kernel wrapper with executable input, architecture and correctness contracts |

## How a run goes

```text
preflight -> tasks, in a fixed order -> certification of the final commit -> verdict and report
```

1. **Preflight** (Python, plus one probe turn per agent configuration): the spec, the checkout, the
   checkpoint, the accuracy anchors, the devices, the agent backends. It compiles the contract. Any
   failure stops the run before a worker starts; the same command resumes once it is fixed.
2. **Tasks**, in this order, which no agent chooses:
   - the survey: `plan.md` and `vocabulary.yaml`, the checkpoint mapped onto the catalog;
   - for each unit, core first, then each feature of the spec: one kernel task per vocabulary
     queue row the unit needs, then the unit itself;
   - one task per configuration variant.

   Tasks run one at a time, each starting from the commits of the tasks before it. Each task
   is one agent turn in its own git worktree. When the worker
   says it finished, its commits are audited (who made them, the sign-off, and that they change
   only the run's own paths) and fast-forwarded into the checkout. **A worker checks its own task**
   with `trtllm-modeling-v2 gate` and its decision is trusted: no check runs between tasks.
3. **Certification**: every check of the contract runs once, on the commit the checkout is at, then
   the delivery checks (the files on disk, the commit audit, the delivered tests, the CI test
   lists). One rule decides what counts: **a verdict counts for the commit it measured and for no
   other.** The checkout must be clean and must not move while the checks run.
4. **Verdict**: `delivered` when all of it passed on that one commit. `delivery_report.md` in the
   workspace says what was certified, on which commit, and what was not.

**The planner** is asked only when the order cannot go on by itself: a worker ended its task
without finishing it, a task spent its attempts (`--max-attempts`), the certification failed, or
you left a message. It answers with one of four decisions: `CONTINUE` (go on; a failed task runs
again with the planner's note), `REPAIR` (run up to three tasks first: a kernel entry or a fix),
`SKIP` (leave out a failed kernel task) or `ESCALATE` (stop). A round after a repair runs the
checks that failed first and stops when one fails again; a run gets `--max-rounds` rounds.

| Verdict | Meaning | Re-running the same command |
|---|---|---|
| `delivered` | every required check and delivery check passed on one commit | nothing to do |
| `escalated` | the planner named a requirement the run cannot meet | re-enters once you fixed it |
| `certification-failed` | the rounds are spent and the final commit is not certified | re-enters |

Exit status: 0 delivered (or a dry run whose preflight passed), 1 any other outcome, 2 a spec or
usage error, 3 the workspace is held by another process.

## Use through your agent

Open Claude Code or Codex in the TensorRT-LLM checkout. Choose the model, reasoning effort and
execution permissions appropriate for your environment, then describe the task. Provide checkpoint,
checkout and workspace paths, the feature and parallelism scope and the accuracy reference.
The agent prepares the spec and launches the workflow; missing inputs must be resolved, not guessed.

**The agent writes the spec by following [spec/INTAKE.md](spec/INTAKE.md)**: `trtllm-modeling-v2 intake
derive` writes a starting spec, and `trtllm-modeling-v2 intake lint` checks it for what would stop
a launch or judge it wrongly (an accuracy check without CUDA graphs, a member too narrow for one
engine, an accuracy anchor that cannot be read or is a placeholder, a requested cell no check
covers, a KV-cache dtype the engine does not load) and prints the contract's checks per unit and
variant. Launch only once the lint is clean or its findings are accepted in `notes`, and after a
`--dry-run`, which touches no GPU.

**Example prompt** — replace placeholders and omit requirements you do not need.

```text
Use the trtllm_modeling_v2 agent flow to onboard <model> from <checkpoint path>.

Requirements:
- Platform: <GPU type and count>; parallelism: <TP / TEP / attention-DP modes and sizes>.
- Workload: <input/output tokens>; weight precision: <dtype>; KV cache: <formats to support>.
- Features: <e.g. core, chunked prefill, block reuse, disagg, multimodality>.
- Exclude: <e.g. speculative decoding>.

Validation:
- Fully validate <features/configurations>; limit <others> to <aggregated text-only / smoke>.
- Accuracy reference: <benchmark and reference/config>. Report unsupported or unverified cases.
```

Roles run on the backend, model and reasoning effort the spec states in `agents.defaults` (the
launching agent writes its own: [spec/INTAKE.md](spec/INTAKE.md), section 7), with per-role
overrides under `agents.roles`; the keys are in [task.example.yaml](task.example.yaml). Nothing
is inherited from the launching session or the environment, and a role whose model the spec does
not state stops the run before any worker starts. Roles load the run's own skills only:
preflight's agent probe reads the skills a session would load and turns off the user's, and Codex
roles run without the user's plugins and apps. Both backends use their native CLI's login,
configuration and state directories from the launching environment. Set up those CLIs before
launching the flow; preflight checks whether they can perform the required work. The workflow
never copies credentials, parses native state configuration or chooses storage locations. It also
leaves sandbox/approval policy to the native CLI, never silently upgrades the runtime on PATH,
and surfaces startup errors to the run's journal. Configure permissions before launching; an
unattended capability probe must complete without interactive approval.

**Compute.** `environment.kind: local`, the one kind: every command, a worker's and the
certification's, runs on the orchestrator's own host, on the devices preflight claims. On a
cluster, start the orchestrator inside the allocation that holds the GPUs, on a host that sees the
checkout, the checkpoint and the workspace. The workflow has no launcher, queue or scheduler of its
own; `infra/compute/executor.py` is the one function to wrap for a package that wants commands to
land elsewhere.

**While it runs**, leave the work to the run. `trtllm-modeling-v2 monitor status --workspace <ws>`
says whether the orchestrator is alive, which tasks ended and how the last certification round
went. To change the direction, send
`trtllm-modeling-v2 feedback --workspace <ws> "<what to change and why>"`: a few sentences naming
any evidence by its path (past 1200 characters the planner reads the head and the path of the
whole). The planner is asked about it between two tasks. A message does not change a check:
`task.yaml` defines the contract.

**If it stops**, run the same command again. It resumes from `tasks.yaml`: a finished task is not
asked again, a task whose worker had already answered keeps that answer, and a certification
round goes on from the verdicts it recorded for the same commit. `--clean` starts over.

## What is where in the workspace

| File | What it is |
|---|---|
| `task.yaml` | the resolved spec every agent reads |
| `contract.yaml` | every check the final commit must pass |
| `tasks.yaml` | the tasks in order, and where each stands: pending, running, done, failed, skipped |
| `decisions/<task>-<attempt>.yaml` | what each turn decided, written by the agent |
| `contract_status.jsonl` | every verdict, with the commit and round it belongs to |
| `certification.yaml` | the last round's result |
| `delivery_report.md` | the verdict, the certified commit, every check, the coverage |
| `journal.md`, `reports/`, `turns/` | what happened, each dispatch and its answer, each turn's activity |

## Code map

| Location | Responsibility |
|---|---|
| `cli.py`, `spec/` | Entry point; task schema, derivation, lint and configuration variants |
| `agents/` | Role and model routing, turns, the decision table, prompts and skills |
| `operators/` | Inventory, prior art and the vocabulary (`vocabulary.yaml`) |
| `run/workflow.py` | Lifecycle: init, resume, turn dispatch, close |
| `run/pipeline.py` | The fixed order of tasks, the planner's four decisions |
| `run/integration.py` | One worktree per task, the commit audit, rebase and fast-forward |
| `run/contract.py`, `run/certify.py` | The contract and its verdicts; the certification, the delivery checks, the report |
| `run/startup/` | Preflight and the agent probe |
| `checks/`, `checks/harness/` | The gates and their evidence; `gate_cli.run_check` runs one for a worker and for the certification alike |
| `infra/` | Command execution on this host, Git operations, the heartbeat and `monitor` |
