<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# trtllm-modeling-v2

Build a TensorRT-LLM Modeling V2 target from a HuggingFace checkpoint. Agents write the code; Python fixes the order of the tasks, lands their commits and runs every check.

**`onboarded`** means: the first deployment of the spec (`core`) boots with `modeling_v2=require` on the commit that landed, and each of its benchmarks meets its frozen floor (`anchor - 2 x SE`, anchors in `<ws>/anchors.yaml`). With no anchor it holds on the boot alone, and the report says `accuracy unjudged (boot only)`. Once met, the flow writes `refs/mv2/<slug>/onboarded` (created once, never moved) and keeps the native libraries its boot measured in `<ws>/onboarded_libraries/`, with a manifest; nothing later takes it back, and a later start that stops early still exits 0 and says in the report why it stopped.

**Exit status:** 0 onboarded (or a `--dry-run` whose preflight passed), 1 not onboarded, 2 a spec, state or usage error, 3 the workspace is held by another process.

## Start

```bash
trtllm-modeling-v2 intake derive --checkpoint <ckpt> --repo <trtllm> --out <spec-dir>   # a first spec
trtllm-modeling-v2 --task <spec-dir>/task.derived.yaml --workspace <ws> --dry-run        # no GPU
trtllm-modeling-v2 --task <spec-dir>/task.derived.yaml --workspace <ws>
```

On one host with a simple environment, fill `checkpoint_path`, `trtllm_repo_path`, `arch`, `environment.python`, `agents.defaults` (`backend`, `model`) and one deployment with its options file. Accuracy runs offline: each benchmark's dataset is under `$LLM_MODELS_ROOT/datasets/` (GSM8K: `openai/gsm8k`), in the HuggingFace cache, or at its `dataset_path`; leave `accuracy` out to onboard on the boot alone. [spec/INTAKE.md](spec/INTAKE.md) says how to add deployments and benchmarks; `task.example.yaml` shows every key.

## How a run goes

1. **Preflight** probes where `core` runs and stops only on a hard blocker: an unreadable `config.json`, a tree that does not import, an unwritable `TMPDIR` (61 characters at most), too few free devices for `core`, a missing dataset of `core`, an agent CLI that does not answer. A config the tree cannot load adds a `config` task first. A later deployment that does not fit, and a benchmark with no dataset or whose base is not attempted, are recorded `not attempted`. A checkout that holds changes or untracked sources no commit has is said at the start: a check that runs in it gives no verdict until they are committed, removed or git-ignored.
2. **Survey** writes `plan.md` and `vocabulary.yaml` (the operators the checkpoint needs, and the kernel rows to add).
3. **Core**, after the kernel rows it needs: each task is one agent turn in its own worktree, 3 attempts, a spent kernel row skipped. When the worker says `ready`, code runs `core`'s boot and benchmarks on its commit; only a pass lands it. A `config`, `survey` or `core` task that spends its attempts ends the run `not onboarded`.
4. **Each further deployment**, in the spec's order. One that spends its attempts is `not done`: nothing of it lands and the run goes on.
5. **Final pass**, once: every check the done deployments owe, the target's tests, and the convention scan (an entry without `CELLS` or `ARCHS`, an import of Modeling V1 model code; reported, never a gate). What fails goes to one `fix` task, then every check of each deployment that had a failure, and `core`'s, run again.
6. **Report**: `<ws>/delivery_report.md`.

A check is `trtllm-modeling-v2 gate <boot|accuracy|tests> --task <ws>/task.yaml --deployment <name> --tree <tree>`, the same command for the run and for a worker. Its verdict line names the directory of its files; `tests` on a deployment that states none says `[NOTHING TO RUN]` and exits 0.

## The report

```text
# onboarded: commit `c46647fa...`, ref `refs/mv2/<slug>/onboarded`, 2026-11-11T16:00:53+00:00
## Done
| deployment | check | score | anchor | verdict | judged | commit | engine attested |
| `tp1` | `accuracy[gsm8k]` | 41.7 | reference 41.77, floor 39.05 (...) | PASS | trtllm-eval | `fa0d912fafd2` | [engine match: 1 rank(s) x 15 option(s); target ...] |
## Not done / Not attempted / Changed outside the target / Skipped kernel rows
## Convention findings left / Final pass / What the run did not check / What the workers left / Models per role / Tasks
```

A run that is not onboarded says in its first lines where it stopped, the result line, the `gate` command that reproduces it and the last commit where that check passed. A score nothing judges is `no verdict`, never a pass or a failure. `engine attested` is every option of the deployment against what each rank's engine resolved; a rank that reported nothing is `unknown`, which fails the check. Each row names the commit it was measured on and the floor its score was held to, and a line under the table names every row that was not measured on the head. "What the workers left" is each task's own `not_done`, a sentence several tasks wrote printed once.

## A multi-node site, and disaggregation

Nothing here names a scheduler. A site adds three fields:

- `launch`, on a deployment or on each of `ctx` and `gen`: the prefix that places the server's ranks (allocation and container only; the flow adds the ranks' launcher). It starts one process per rank, which the flow checks before a load; a prefix that starts one process per node and makes the ranks inside it is not supported. It stays in the foreground, sees the same workspace, checkout and checkpoint paths, and gives rank 0 a hostname this host reaches. `ctx` and `gen` prefixes place their ranks on different nodes of the allocation (two prefixes that each say "four nodes" may name the same four); the proxy runs on the orchestrator's host. `environment.launch` is the prefix of one-GPU commands: an entry's test in a worker's hands, and the target's delivered tests.
- `stop`, beside `launch`: the site's command that ends what `launch` started. Both receive `MV2FLOW_SERVER=<deployment>[.<role>]`.
- `env`, on a deployment: what its servers need of the site.

A disaggregated deployment (`ctx` and `gen`) states `cache_transceiver_config` with a `backend` in both sides' options (`{backend: DEFAULT}` lets the engine choose; a spec without it is refused at load), and its `env` states the site's UCX transports: a container's `UCX_TLS=tcp` stops the NIXL backend from starting, and `UCX_TLS: "^ib"` worked on GB300.

## Resume, notes, spec edits

- **Resume:** the same command against the same workspace goes on from where the last start stopped. A cut turn restarts, free twice. Each new start gives tasks not done fresh attempts: 4 times for `config`, `survey` and `core`, 2 for any other. `--clean` starts over; `trtllm-modeling-v2 monitor --workspace <ws> status` (or `wait`) follows a run.
- **A note:** `trtllm-modeling-v2 feedback --workspace <ws> "<note>"` (or `--file <path>`) appends to `<ws>/operator_notes.md`. Every later turn reads it first and again before its receipt. To reach the turn in progress, stop the run and start it again. A note never permits changing an anchor, a tolerance or a check.
- **The spec:** add deployments to `task.yaml` and start again. The first deployment stays first.

## What the flow does not check

- Performance: there is no tuning stage, no sweep and no throughput claim.
- Whether the target's tests are registered with the repository's CI.
- Whether a disaggregated generation server reused what its context server computed: the proxy does not show it.
- Commits on the branch that no task of this run landed.
- Conventions are findings in the report, not gates.

Run on GPUs so far: every check once, standalone, on one node (aggregated and disaggregated, GSM8K). Two ranks per server have run (aggregated and disaggregated). Not yet: a prefix that leaves the host, `stop`, a restart after a killed process, image prompts. Through a proxy only GSM8K is scored; another benchmark there is refused, not judged.

## An agent that operates a run

An outside agent may drive a run without a person: change the environment, the spec, shared code or the target, send notes, and start again within the bounds above. Two rules: every change is a commit or a spec edit, so the report lists it; and it never alters an anchor, a tolerance or a check. A commit that touches `tensorrt_llm/evaluate/`, `tensorrt_llm/commands/eval.py` or the accuracy references is refused.
