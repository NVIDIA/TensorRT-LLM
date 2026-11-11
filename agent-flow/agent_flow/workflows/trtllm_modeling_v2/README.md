<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# trtllm-modeling-v2

Onboard a HuggingFace checkpoint as a TensorRT-LLM ModelingV2 target: reuse or certify catalog
operators, assemble the model, bring up its first deployment and then each further one the spec
states. Agents write code; Python owns the order of the work, the landing of each task, the checks
and the report. Performance tuning is another tool's work, on the onboarded target.

## Terms

| Term | Meaning |
|---|---|
| Target | Model implementation for a checkpoint family and the deployments it supports |
| Deployment | One way the target is served: its engine options, its launch, its benchmarks. The first is `core` |
| Check | One gate (boot, accuracy, serve, tests) of one deployment |
| Task | One piece of work for one agent turn: the survey, a kernel entry, a deployment, a fix |
| Entry | A kernel wrapper with executable input, architecture and correctness contracts |
| Onboarded | The first deployment's boot and benchmarks passed, by code, on the commit that landed |

## How a run goes

```text
preflight -> tasks in a fixed order, the onboard check after core -> the final pass -> the report
```

1. **Preflight** (Python, plus one probe turn per agent configuration). It stops the run before
   a worker starts only on a hard blocker: the checkpoint cannot be read, the tree does not import,
   the first deployment's devices are missing, busy or too small for its weights, its temporary
   directory is not writable, a benchmark it is scored on has no dataset, an agent does not answer.
   The tree, the devices and the temporary directory are probed where the first deployment runs
   (behind its launch prefix). Everything else it sees is a note; a later deployment that cannot be
   attempted, and a checkpoint config the tree cannot load yet, are recorded in `preflight.json`.
   It freezes the accuracy anchors (`anchors.yaml`). The same command starts again once a blocker
   is fixed.
2. **Tasks**, in this order, which no agent chooses:
   - a config task, only when the tree cannot load the checkpoint's config;
   - the survey: `plan.md` and `vocabulary.yaml`, the checkpoint mapped onto the catalog;
   - `core`, the first deployment, after one kernel task per vocabulary queue row it needs;
   - each further deployment in the spec's order, after the kernel rows it needs.

   Tasks run one at a time, each starting from the commits of the tasks before it. Each task
   is one agent turn in its own git worktree. When the worker says it is ready, its commits are
   audited (who made them, the sign-off, and that none touches the workspace or a judge path),
   rebased, fast-forwarded into the checkout and labelled `target`, `shared` or `mixed` by whether
   their paths are the target's own. Each landing is read by the conventions scan (an entry's
   `CELLS`, `ARCHS` and test, Modeling V1 imports, the landing name): its findings are kept for
   the report and never hold a task.
3. **The onboard check**, by code: when `core` says it is ready, the run boots and scores the first
   deployment under `modeling_v2=require` on the worker's clean, committed worktree, against the
   frozen anchors. A pass lands that commit, records `onboarded` (the commit, the content hash of
   its native libraries, the time) and points `refs/mv2/<slug>/onboarded` at it; nothing later
   takes that back. A failure is the attempt's result line and the next attempt's note. A
   benchmark nothing anchors is a score with no verdict, and with none judged the record says
   `accuracy unjudged (boot only)`.
4. **The final pass**, once, after the last task: the remaining checks of every done deployment,
   the target's own tests and the commit audit, each a report line naming the commit it was
   measured on, the path that judged it and what it attested of the engine. A pass on record for
   the same content is reused, and a deployment whose boot failed is not scored. When something
   failed or the conventions scan has findings, one `fix` task names them all; then each
   deployment that had a failure and core's own checks run once more.
5. **The report**, `delivery_report.md`: its first line says `onboarded` (commit, ref, time) or
   `not onboarded` (where it stopped, the gate command that shows it again, the last commit where
   that check passed); then what is done, not done, not attempted, changed outside the target, the
   skipped kernel rows, the findings left, what the run did not check and each role's model.

**The bounds are numbers.** A task gets three attempts (`--max-attempts`); a failed attempt runs
again with its result line as its note. An attempt whose turn a stopped process cut restarts free
twice. A task that spent its attempts is not done: nothing of it lands and the run goes on, except
for the config task, the survey and `core`, where the run ends not onboarded. Each new start
reopens a task not done with fresh attempts, four times for those three and twice for any other.

Exit status: 0 onboarded, whatever else is not done (or a dry run whose preflight passed), 1 any
other outcome, 2 a spec or usage error, 3 the workspace is held by another process.

## Use through your agent

Open Claude Code or Codex in the TensorRT-LLM checkout. Choose the model, reasoning effort and
execution permissions appropriate for your environment, then describe the task. Provide checkpoint,
checkout and workspace paths, the feature and parallelism scope and the accuracy reference.
The agent prepares the spec and launches the workflow; missing inputs must be resolved, not guessed.

**The agent writes the spec by following [spec/INTAKE.md](spec/INTAKE.md)**: `trtllm-modeling-v2 intake
derive` writes a starting spec and prints three warnings where they apply: a first deployment
too narrow for the checkpoint's weights, a first deployment with no accuracy anchor, and a
deployment scored without CUDA graphs. Launch after a `--dry-run`, which validates the spec,
journals the same warnings and the deployments the spec comes to, and touches no GPU.

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
run's own checks, runs on the orchestrator's own host, on the devices preflight claims. On a
cluster, start the orchestrator inside the allocation that holds the GPUs, on a host that sees the
checkout, the checkpoint and the workspace. The workflow has no launcher, queue or scheduler of its
own; `infra/compute/executor.py` is the one function to wrap for a package that wants commands to
land elsewhere.

**While it runs**, leave the work to the run. `trtllm-modeling-v2 monitor status --workspace <ws>`
says whether the orchestrator is alive (from any node that sees the workspace), whether the run is
onboarded and where its tasks stand. To tell the workers something, run
`trtllm-modeling-v2 feedback --workspace <ws> "<what, and why>"`: the text is appended, with its
time, to `operator_notes.md`, which every later turn is told to read first and again before it
decides. No agent interprets it and it adds no task; to reach the turn in progress, stop the run
and start it again. A note never changes an anchor or a check: the spec defines those, and it is
read again on every start, so a new deployment is an edit of the spec.

`core` is the first deployment of the spec: a start whose first deployment is another one than
`core` was built for is refused. Add deployments after it; to onboard another first, use another
workspace.

**If it stops**, run the same command again. It goes on from `tasks.yaml`: a done task is not
asked again, a task whose worker had already answered keeps that answer, a cut turn restarts, a
task not done gets fresh attempts within its grants, and a check that passed on the same content
is not run again. `--clean` starts over; the frozen anchors and your notes stay.

## What is where in the workspace

| File | What it is |
|---|---|
| `task.yaml` | the resolved spec every agent reads |
| `tasks.yaml` | the tasks in order, and where each stands: pending, running, done, not done, skipped |
| `decisions/<task>-<attempt>.yaml` | what each turn decided, written by the agent |
| `contract_status.jsonl` | every verdict, with the commit and the content it measured |
| `delivery_report.md` | whether the run is onboarded, and every deployment where it stands |
| `operator_notes.md` | what `feedback` appended, for the workers |
| `anchors.yaml`, `preflight.json` | the frozen anchors; what preflight left out and why |
| `conventions.json`, `commit_labels.json` | the scan's findings by first task; each landed commit's label |
| `journal.md`, `turns/` | what happened; each turn's prompt, activity and answer |

## Code map

| Location | Responsibility |
|---|---|
| `cli.py`, `spec/` | Entry point; task schema, deployments, derivation and configuration variants |
| `agents/` | Role and model routing, turns, the decision table, prompts and skills |
| `operators/` | The vocabulary (`vocabulary.yaml`) and the inventory of the tree's operators |
| `run/workflow.py` | One start of a run: the spec, the state, the agents, the report |
| `run/pipeline.py` | The fixed order of tasks, attempts, cut turns, grants, landing |
| `run/integration.py` | One worktree per task, the commit audit, rebase, the landing check, fast-forward |
| `run/contract.py`, `run/report.py` | The checks each deployment owes and their verdicts; the onboard check, the final pass, the report |
| `run/startup/`, `run/start.py` | Preflight's hard blockers and the agent probe; what a run starts from |
| `checks/`, `checks/harness/` | The gates and their evidence; `gate_cli.run_check` runs one for a worker and for the run alike |
| `infra/` | Command execution on this host, Git operations, the heartbeat and `monitor` |
