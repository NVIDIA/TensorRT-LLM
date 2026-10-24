<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mechanisms: what each one is for, and what keeps it

Each mechanism with the failure it exists for (a [LESSONS](../LESSONS.md) number where a run
produced it), its switch and default, and whether a GPU run has exercised it. A default-on
mechanism names a failure it prevents; one no run has exercised stays off by default until one has
(except primers, below); one that runs for several runs without catching anything is a candidate to
remove. `metrics.json` (`run/metrics.py`) gives the evidence: per gate kind, how often it ran,
failed and said nothing, and the device-hours it took.

## Correctness: on by default

| mechanism | for | switch | GPU-exercised |
|---|---|---|---|
| contract of checks, judged by Python on the tree they measured (`run/contract.py`) | agents claiming done on evidence that was not there (15, 16, 32) | always | yes |
| boot and accuracy at every member, `modeling_v2=require` (`checks/gates.py`) | a configuration that silently served the built-in model (16) | always; `profile: bringup` takes one member per unit | yes |
| acceptance gate for a speculator (`checks/gates.py`) | a miscomputed draft path that boot and accuracy cannot see (18) | on for speculating units | yes |
| statistical accuracy floor (`checks/stats.py`) | a member-wide shift inside the tolerance (19) | `accuracy.floor` | yes |
| read-only snapshots, verdict fingerprints and read sets (`run/snapshots.py`, `run/workflow.py`) | a verdict measured on a tree another writer was changing (15, 32) | always | yes |
| survey, vocabulary and mechanical kernel fan-out (`onboard/`) | a planner turn spent on what the survey already decided (7, 10, 11) | always | yes |
| local variants and the promote (`onboard/promotion.py`) | tasks on one shared entry serialized on one file (29) | always | yes |
| trunk integration, rebase then fast-forward (`run/integration.py`) | an agent turn spent on every join | always | yes |
| regression check with no agent turn (`run/regression.py`) | a shared change that broke another model's package tests | after each landed shared change | CPU only: the test cache it runs through was exercised on GPU runs, the agentless check was not |
| per-cell delivery and the `-best` branch (`run/workflow.py::_mark_cells`) | a run stopped with 93 of 139 checks green and nothing to deliver | always | CPU only |
| run budget (`run/metrics.py`) | a run with no bound on wall clock or device-hours | `budget` in `task.yaml`; none by default | CPU only |
| operator inbox (`run/inbox.py`) | an operator's correction that could reach the run only through a reload | `trtllm-modeling-v2 feedback` | CPU only |
| job priority classes (`infra/execution/priority.py`) | the first accuracy result queued behind feature checks | always | CPU only |
| process-group end of a timed-out local command (`infra/execution/commands.py`) | a launcher's ranks outliving their timeout and holding devices (37) | always | CPU only |

## Proofs a contract can add

| mechanism | for | switch | GPU-exercised |
|---|---|---|---|
| witnesses (`checks/witness.py`) | a feature that boot and accuracy pass whether or not it works (21, 22, 23) | `accuracy.witness.enabled`, off; on under `profile: certified` | yes |
| envelope boot (`checks/envelope_boot.py`) | a target that fails at its declared limits and nowhere the gates went (11) | `accuracy.envelope.enabled`, off; on under `profile: certified` | yes |
| serve gate (`checks/serve_gate.py`) | the server path differing from the LLM API's | `accuracy.serve.enabled`; on where the spec bounds the server's length, and under `profile: certified` | yes |
| logit gate (`checks/logit_agreement.py`) | a member whose output no benchmark judges | `accuracy.logits` | yes |

## On before a GPU run has exercised it

| mechanism | for | switch | why it is on |
|---|---|---|---|
| primers: turns forked from one briefing session per role (`run/primers.py`) | every turn of a role reading the same files again | on; `--no-primers` turns them off | it falls back rather than fails: a primer not built yet, or that failed to build, changes no turn; a fork that fails drops the session, and the turn runs again pointed at the primer's summary; a forked turn is told the commit its reading was done at, and diffs a file against it before relying on it |

## Off until a GPU run has exercised it

| mechanism | for | switch |
|---|---|---|
| managed sessions (`infra/substrate/managed.py`) | sessions started and handed over by hand | `--manage-sessions` |

## Open

- A cheap performance gate (20): a prefill chunk and a decode step against the previous tree.
- A primer for the tuner's roadmap turns, once a GPU run has exercised primers.
- Retire the gates whose `metrics.json` rows show no failure across several runs of different
  models.
