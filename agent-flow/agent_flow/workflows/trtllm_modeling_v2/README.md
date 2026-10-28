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

## Design

1. **Specify and prepare:** derive or validate `task.yaml`, check the environment and agent access,
   survey available operators, and compile a contract of required checks. A unit is core or a
   feature; a member is a parallel configuration; their pairing defines a coverage cell.
2. **Plan and execute:** the planner proposes a dependency graph; Python admits and schedules it.
   Workers use separate worktrees. Kernel tasks certify or add operators; assemblers build the
   target. Commits are audited and integrated; shared-code changes require regression coverage.
3. **Prove and deliver:** checks measure fixed-commit snapshots against framework-owned references.
   Evidence is reused only while its relevant code, inputs and environment remain valid. Missing
   engine facts stay unknown. Delivery requires the contract, tests, files and promotion to agree.
4. **Resume:** checkpoints, decision identities and task receipts recover completed work without
   adopting another attempt's answer. Compute sessions own resource allocation and safe cancellation;
   the operator handles cluster allocations and session renewal.
5. **Tune optionally:** tuning can start after core checks pass or run independently. Tuner proposals
   run in parallel; verification and keeps currently serialize. Local speedups need final cumulative
   end-to-end validation. Results stay on a separate best branch, with `tune/summary.md`,
   `perf_ledger.jsonl` and `UNMERGED_OPTIMIZATIONS.md`; they are not merged into delivery automatically.

Current tuning retains configurable smoke, stock reference and noise repeats. A strict single
baseline and parallel candidate measurement remain planned changes. Disaggregation currently uses a
supplied contrast leg; through-the-hop accuracy is not implemented. Engine attestation still needs
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
