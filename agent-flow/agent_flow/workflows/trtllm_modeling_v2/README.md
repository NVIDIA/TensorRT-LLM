<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# trtllm-modeling-v2

Build a TensorRT-LLM ModelingV2 target from a HuggingFace checkpoint: reuse or certify catalog
operators, assemble the model and prove the requested features. Performance tuning is not part of
the flow: it runs on the delivered target, with the tools of your choice.
Agents write code; Python owns scheduling, integration and correctness verdicts.

## Terms

| Term | Meaning |
|---|---|
| Target | Model implementation for a checkpoint family and supported deployment configurations |
| Unit | Core implementation or one feature added to it |
| Member | One parallel configuration, such as `tp4`, `tep4` or `dep4` |
| Coverage | Each requested unit or variant, with the required checks behind it |
| Entry | A kernel wrapper with executable input, architecture and correctness contracts |
| Receipt | A task's recorded decision, commits and checks |
| Session | A process on a compute node that watches a spool and runs its queued commands |

## Use through your agent

Open Claude Code or Codex in the TensorRT-LLM checkout. Choose the model, reasoning effort and
execution permissions appropriate for your environment, then describe the task. Provide checkpoint,
checkout and workspace paths, compute configuration, feature/parallelism scope and accuracy reference.
The agent prepares the spec and launches the workflow; missing inputs must be resolved, not guessed.

**The agent writes the spec by following [spec/INTAKE.md](spec/INTAKE.md)**: `trtllm-modeling-v2 intake
derive` writes a starting spec (no `site.yaml` needed). The spec should be the smallest contract
that covers the request: one check of any kind covers a feature, and `--dry-run` journals the
contract's checks per unit and variant before anything is spent. Preflight reads the checkpoint and the spec and loads no model. On a `local`
host it also claims the devices and runs one NCCL collective across the ranks; on a session the
allocation is the claim and nothing probes its host. A host fault appears at the first load,
where the first thing to settle is whether it is the host's or the target's.

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
and surfaces startup errors to the workflow's recovery logic. Configure permissions before
launching; an unattended capability probe must complete without interactive approval.

**Compute.** `environment.kind: local` runs every job on the orchestrator's own host and devices.
`session_spool` runs GPU jobs through compute sessions: processes on GPU nodes, each running the
run's spool scheduler on a directory, started however your cluster starts work (a batch scheduler,
a container runtime, a shell). The run needs:

- each session to run its entry script, `trtllm-modeling-v2 session render --task task.yaml
  --name <session>` (render it again after changing `environment.env`);
- a session to stay up while the run needs it: jobs queued on a session that is gone wait
  `environment.session_wait`, then come back without a verdict;
- a session with a `time_limit` to be followed before its end: from `handover_lead` before then it
  starts only jobs that fit, and the next session started on the same spool takes the rest;
- one orchestrator per workspace (a new one resumes from it), on a host that sees the run
  directory, checkout and checkpoints at the same paths as every session.

`trtllm-modeling-v2 monitor status` shows each session, and raises `SESSION_ENDING` while one is in
hand-over with no newer session on its spool. A gate or test job the run has
waited on for 15 minutes writes a `waiting` event: the job, its spool, queued or running and since
when, why the session holds it when its `status.json` says, and for a queued job what runs on its
spool meanwhile; then at most one every 15 minutes while the wait lasts.

**While it runs**, you or the launching agent keep the sessions alive and leave the work to the run. Never cancel
a job the run submitted (a gate, a worker's test) or run a replacement check beside it: a cancelled
check proves nothing, runs again, and the planner never learns why. To change the direction, send
`trtllm-modeling-v2 feedback --workspace <ws> "<what to change and why>"`: a few sentences naming
any evidence by its path, not status reports (past 1200 characters the planner reads the head and
the path of the whole). A message rides with the planner's next turn; `--now` wakes it, for what
must change before that.

## Design

1. **Specify and prepare:** derive or validate `task.yaml`, check the environment and agent access,
   survey available operators, and compile a contract of required checks. A unit is core or a
   feature; a member is a parallel configuration. One check of any kind covers a feature.
2. **Plan and execute:** the planner proposes a dependency graph; Python admits and schedules it.
   Workers use separate worktrees. Kernel tasks certify or add operators; assemblers build the
   target. Commits are audited and integrated. What a change to shared code does to other
   models is the pull request's CI to show, not the run's. A kernel task on a shared
   catalog entry lands a local variant in the target's own `ops/`, and the target is delivered
   with it: the run never changes a shared entry, so other models are not disturbed mid-run.
3. **Prove and deliver:** checks measure fixed-commit snapshots against framework-owned references.
   Three rules decide when a check runs:
   - *A pass is kept.* Every required check passes once. A pass does not run again because
     other code changed; a check that has not passed (it failed, or no landing ran it) runs
     again on what the work left.
   - *Whoever lands code names its checks.* A task's `satisfies` is the planner's guess; its
     worker answers with the checks of `contract.yaml` its change can affect, and they run on
     what it landed, passed before or not. A run that gives no result leaves them owed.
   - *The final code is scored once.* Before delivery a benchmark has passed on the final
     tree; if none has, the cheapest one runs.

   The report names the commit each check measured. Delivery requires the contract, tests
   and files to agree.
4. **Resume:** checkpoints, decision identities and task receipts recover completed work without
   adopting another attempt's answer. Compute sessions own resource allocation and safe cancellation;
   you or the launching agent handle cluster allocations and session renewal.

## Code map

| Location | Responsibility |
|---|---|
| `cli.py`, `spec/` | Entry point; task schema, derivation and configuration variants |
| `agents/` | Role/model routing, turns, slots, decision protocols, prompts and entry skill |
| `operators/` | Inventory and operator coverage (`vocabulary.yaml`) |
| `run/workflow.py`, `orchestrator.py`, `graph.py`, `inbox.py` | Lifecycle; scheduling; graph rules; operator input |
| `run/startup/` | Bootstrap, preflight, agent probes and role primers |
| `run/execution/` | Task dispatch, kernel-specific preparation and decision/receipt handoffs |
| `run/validation/` | Contract, check runner, test jobs, completion and coverage |
| `run/workspace/`, `run/records/` | Worktrees/snapshots; recovery state, journal and metrics |
| `checks/`, `checks/harness/` | Verdict rules and evidence; programs executed in the measured environment |
| `infra/compute/`, `infra/monitor/` | Execution, sessions, scheduler and provenance; status and heartbeat |
| `infra/git.py`, `infra/files.py` | Shared Git/worktree operations and atomic file I/O |
