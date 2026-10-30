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

Role models are resolved once per launch: per-role settings override global settings, then explicit
backend model environment overrides, then the launching session. Claude uses its last recorded
assistant model, which can lag a switch; Codex queries its live thread. If lookup is unavailable,
set the model explicitly. Custom models and effort remain available through `agents.defaults` and
`agents.roles` in [task.example.yaml](task.example.yaml). Model inheritance does not grant permissions.
Roles load the run's own skills only: preflight's agent probe reads the skills a session would load
and turns off the user's, and Codex roles run without the user's plugins and apps.

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
5. **Tune optionally:** tuning is background work for what the onboarding's workers leave, and
   never holds the onboarding back. It starts once the core's required checks have passed (the
   features' and variants' checks go on beside it; delivery still owes them), or runs on its own
   (`trtllm-modeling-v2 tune`). Its spool jobs queue below every job of the onboarding and never
   rise above them while they wait; it reserves none of the run's devices; its turns take at most
   two agent slots while the onboarding has work open. The onboarding's verdict (delivered,
   escalated, ...) closes the run when it comes; a campaign still running goes on past it as the
   tune stage, and ends in `state.tune["verdict"]`, never in the run's verdict; a process that
   stops meanwhile resumes it with the same command, or `tune`. It measures one baseline, with no
   smoke, stock reference or repeat, then starts the campaign. Tuner
   proposals run in parallel; verification and keeps serialize. A micro-bench speedup is projected
   to end-to-end through the share of the step its kernels take, its outputs held roughly to the
   control's, and a win keeps at once. Every few keeps a checkpoint measures end-to-end, the first
   output token's distribution and the target's suite, and a failure rolls those keeps back; the
   accuracy benchmark runs once, on the final tree. Perf points default to concurrency 1/32/128,
   an attention-DP member's to the high point. The speed of light comes from independent
   analyses (config formulas, the code, a profile), reconciled: what they dispute is ruled on
   once, by choosing among them, or stays unknown. Every finished attempt, failed or not, reaches
   the later requests with the stage's verdict and the turn's findings (`tune/attempts.md`), and
   a loss the code can fix goes back to its turn once. Results stay on a separate best branch, with `tune/summary.md`,
   `perf_ledger.jsonl` and `UNMERGED_OPTIMIZATIONS.md`; they are not merged into delivery automatically.

Parallel candidate measurement remains a planned change. Disaggregation is proven by the
tree's own disaggregated-serving accuracy test for the target (`accuracy.disagg.test`), run as CI
runs it; the flow deploys nothing of its own. Engine attestation still needs
an internal rank hook where public APIs provide no resolved facts.

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


For a native disaggregated test spanning multiple nodes, set
`accuracy.disagg.nodes` and declare a gate session with at least that many nodes.
`accuracy.disagg.devices` is the GPU count per node. The session spool reserves
whole nodes of one allocation and runs one pytest process on each, exposing
`MV2FLOW_JOB_NODE_RANK`, `MV2FLOW_JOB_HOSTS`, and `MV2FLOW_JOB_NODES` for the
native test's role placement. Every node must write a fresh passing JUnit report
and exit successfully; missing, skipped, or failed parts reject the gate. Engine
TP/EP/attention-DP mappings remain the native test's explicit contract.

`accuracy.timeout` bounds a complete benchmark command (default four hours).
Set it and `accuracy.disagg.timeout` below the compute allocation's remaining
lifetime, including loading and teardown margin. These time limits do not
change the evaluator's sample count or request protocol.

For GSM8K, `--log_samples` also gets a unique `--output_path` below that gate's
work directory unless the task explicitly supplies one. The saved lm-eval JSON
retains prompts, responses, labels and both scoring filters for independent audit.

The same GSM8K runs also save inference text and token IDs under `inference/`
so prompt/output token counts can be audited; an explicit `--output_dir` is preserved.
