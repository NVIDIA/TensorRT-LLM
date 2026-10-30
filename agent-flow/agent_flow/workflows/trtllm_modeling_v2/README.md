<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# trtllm-modeling-v2

Build a TensorRT-LLM ModelingV2 target from a HuggingFace checkpoint: reuse or certify catalog
operators, assemble the model, prove the requested features, and optionally optimize performance.
Agents write code; Python owns scheduling, integration and correctness verdicts.

## Use through your agent

Open Claude Code or Codex in the TensorRT-LLM checkout. Choose the model, reasoning effort and
execution permissions appropriate for your environment, then describe the task. Provide checkpoint,
checkout and workspace paths, compute configuration, feature/parallelism scope and accuracy reference.
The agent prepares the spec and launches the workflow; missing inputs must be resolved, not guessed.

**The agent writes the spec by following [spec/INTAKE.md](spec/INTAKE.md)**: `trtllm-modeling-v2 intake
derive` writes a starting spec (no `site.yaml` needed), and `trtllm-modeling-v2 intake lint` checks
it for slow or stalling choices (eager gates, a batch below the claimed concurrency, a member too
narrow for one engine, disaggregation wider than a session, variant cross products, variants that
boot at every member, a feature proven
at several members or again under another axis, unusable accuracy anchors, open-ended boot probes) and blocks an accuracy check that runs without CUDA graphs and
what would stop preflight (uncovered requested cells, a KV-cache dtype the engine does not load),
and prints the contract's checks per unit and variant, the
engine's constraints for this geometry and the prior art the tree already holds. The spec should be
the smallest contract that covers the request. Launch only once the lint is clean or its findings
are accepted in `notes`. Preflight's substrate gate boots a GPU beside the run's first turns; its
failure stops the run, which the same command resumes once the host is fixed.

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

Tuning (optional):
- Optimize <configurations> for low latency at concurrency <list>.
- Optimize <configurations> for high throughput at concurrency <list>.
- Budget: <time/resource limit>. Preserve correctness and report baseline versus final results.
```

Roles run on the launching agent's own harness, model and reasoning effort, which it writes into
`agents.defaults` ([spec/INTAKE.md](spec/INTAKE.md), section 7): written values hold on any host.
Unwritten, they are resolved once per launch: per-role settings override global settings, then
explicit backend model environment overrides, then the launching session, which never gives the
effort and must be within the orchestrator's reach: its session variables, and Claude's transcript
(its last recorded assistant model, which can lag a switch) or Codex's live app-server, on its own
host only. The keys are in
[task.example.yaml](task.example.yaml). Model inheritance does not grant permissions.
Roles load the run's own skills only: preflight's agent probe reads the skills a session would load
and turns off the user's, and Codex roles run without the user's plugins and apps. Codex roles keep
their state in the run's own home, `<workspace>/codex-home` (the login and configuration copied from
`$CODEX_HOME`), and its SQLite databases on the host's local disk: runs never share them.

**While it runs**, the operator keeps the sessions alive and leaves the work to the run. Never cancel
a job the run submitted (a gate, a worker's test) or run a replacement check beside it: a cancelled
check proves nothing, runs again, and the planner never learns why. To change the direction, send
`trtllm-modeling-v2 feedback --workspace <ws> "<what to change and why>"`: a few sentences (at most
1200 characters) naming any evidence by its path, not status reports. A message rides with the
planner's next turn; `--now` wakes it, for what must change before that.

## Design

1. **Specify and prepare:** derive or validate `task.yaml`, check the environment and agent access,
   survey available operators, and compile a contract of required checks. A unit is core or a
   feature; a member is a parallel configuration; their pairing defines a coverage cell.
2. **Plan and execute:** the planner proposes a dependency graph; Python admits and schedules it.
   Workers use separate worktrees. Kernel tasks certify or add operators; assemblers build the
   target. Commits are audited and integrated; shared-code changes require regression coverage.
   A regression check runs only when no contract check runs or waits, and only on the test files
   the change can reach (those it changed, those importing a module it changed, the claims
   gates; every file when it touches code the whole package shares). Promoting the target's
   local entries into the catalog is the last change: it starts once the contract holds, its one
   check is a boot of the core, and the verdicts from before it hold after it.
3. **Prove and deliver:** checks measure fixed-commit snapshots against framework-owned references.
   Evidence is reused only while its relevant code, inputs and environment remain valid. Missing
   engine facts stay unknown. Delivery requires the contract, tests, files and promotion to agree.
4. **Resume:** checkpoints, decision identities and task receipts recover completed work without
   adopting another attempt's answer. Compute sessions own resource allocation and safe cancellation;
   the operator handles cluster allocations and session renewal.
5. **Tune optionally:** background work for the performance the onboarding's workers leave; it
   never holds the onboarding back (its jobs queue below the onboarding's, it reserves none of the
   run's devices, it takes at most two agent slots while onboarding work is open, and no onboarding
   verdict waits for it). It starts once the core's required checks pass, or on its own
   (`trtllm-modeling-v2 tune`): one baseline, parallel candidates kept one at a time, a checkpoint
   every few keeps that rolls them back on a loss, and the accuracy benchmark once on the final
   tree. Results stay on a separate best branch with `tune/summary.md`; nothing merges into
   delivery, and the outcome is `state.tune["verdict"]`. `tune/stage.py` and `tune/campaign.py`
   describe the policy.

## Code map

| Location | Responsibility |
|---|---|
| `cli.py`, `spec/` | Entry point; task schema, derivation and configuration variants |
| `agents/` | Role/model routing, turns, slots, decision protocols, prompts and entry skill |
| `operators/` | Inventory, operator coverage (`vocabulary.yaml`) and promotion |
| `run/workflow.py`, `orchestrator.py`, `graph.py`, `inbox.py` | Lifecycle; scheduling; graph rules; operator input |
| `run/startup/` | Bootstrap, preflight, agent probes and role primers |
| `run/execution/` | Task dispatch, kernel-specific preparation and decision/receipt handoffs |
| `run/validation/` | Contract, check runner, regression, completion and verified runtime summaries |
| `run/workspace/`, `run/records/` | Worktrees/snapshots; recovery state, journal and metrics |
| `checks/`, `checks/harness/` | Verdict rules and evidence; programs executed in the measured environment |
| `infra/compute/`, `infra/monitor/` | Execution, sessions, scheduler and provenance; status and heartbeat |
| `infra/git.py`, `infra/files.py` | Shared Git/worktree operations and atomic file I/O |
| `tune/`, `tune/harness/` | Campaign, measurement and keep policy; standalone benchmark programs |
