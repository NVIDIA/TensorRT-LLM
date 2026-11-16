<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# trtllm-modeling-v2

Build a TensorRT-LLM ModelingV2 target from a HuggingFace checkpoint and prove each deployment
the spec states. Agents write code; Python owns the task list, the GPU pool, integration and
the checks.

```
intake -> start probes -> survey (typed rows) -> [ kernel rows | weights | routing | state ]  side by side
       -> assemble -> the first deployment's session  => "onboarded" (a pinned ref)
       -> [ each further deployment: its tasks, then its session ]
       -> delivery checks -> report
```

## Install

Python 3.11 or newer: `pip install -e <checkout>/agent-flow` gives the `trtllm-modeling-v2`
command. The engine is built and installed beforehand in an interpreter of its own, which the
spec names (`environment.python`); a run builds no native code. Install Claude Code or Codex
and sign in on the host the command runs on: the start runs each role's agent once,
unattended, and stops when one cannot work.

## What you must write

Everything else has a default or is derived. `intake derive` writes the first four rows from a
checkpoint and a goal; `intake lint` and the start name each key that is still missing, with
how to find its value.

| Key | What it is | Where the value comes from |
|---|---|---|
| `checkpoint_path` | the HuggingFace checkpoint directory | where you downloaded it |
| `trtllm_repo_path` | the TensorRT-LLM checkout the target is built in | your clone, with its build |
| `arch` | the GPU architecture entries are certified on | `sm_100` for B200, `sm_103` for GB300; never inferred |
| `deployments` | each configuration to serve: a name and the options `trtllm-serve --config` takes | the goal's layouts and features |
| `environment.python` | the interpreter of the engine's install | the checkout's virtual environment, or the image's |
| `environment.env.TMPDIR`, `CLAUDE_CODE_TMPDIR` | scratch outside `/tmp`, at most 61 characters | a directory on local or shared scratch |
| `agents.defaults` | the backend, model and reasoning effort every role runs on | your own session's; nothing is inherited |
| `accuracy.model_key`, or a benchmark's `anchor` | what a score is held to | the model's `org/name` when the checkout has a reference row for it; else a measured score with its `source`; else `benchmarks: []` on the deployment |
| on Slurm: `environment.slurm.account`, `.partition`, `.time`, `.gpus_per_node` | where the run asks for its allocations | your site; with a `container_image`, also its `container_mounts` |

## A first run on one host

One host with its GPUs and no Slurm states nothing about compute.

```bash
trtllm-modeling-v2 intake derive --checkpoint <checkpoint dir> --repo <TensorRT-LLM checkout> \
    --goal "tp1 arch=sm_103 model_key=<org>/<model>" --out spec
```

It writes `spec/task.derived.yaml` and `spec/engines/tp1.yaml` and prints what is open. Add
what it cannot know, and save the file as `spec/task.yaml`:

```yaml
environment:
  kind: local
  python: /path/to/TensorRT-LLM/<venv>/bin/python
  env: {TMPDIR: /path/to/scratch/tmp, CLAUDE_CODE_TMPDIR: /path/to/scratch/tmp}
agents:
  defaults: {backend: claude-code, model: <your model id>, reasoning_effort: <your effort>}
```

```bash
trtllm-modeling-v2 intake lint --task spec/task.yaml               # the advice; exit 0 when the spec loads
trtllm-modeling-v2 --task spec/task.yaml --workspace ws --dry-run  # the host, its devices, the agents
trtllm-modeling-v2 --task spec/task.yaml --workspace ws            # the run
trtllm-modeling-v2 monitor --workspace ws wait                     # in another shell: what needs you
```

A goal is words, not a sentence: layouts (`tp1`, `tep4`, `dep4`), features, and `key=value`
(`kv=bf16/fp8`, `isl=`, `osl=`, `concurrency=`); a word it does not know is refused with the
words it knows. `intake derive` writes its files anew each time, `engines/` included: derive
once, then edit. Several deployments, anchors and the notes every role reads are
[spec/INTAKE.md](spec/INTAKE.md); every key with its default is
[task.example.yaml](task.example.yaml).

## On Slurm

Remove the `kind` line and state `environment.slurm` with the four keys above: a stated
`kind: local` wins over a `slurm` block beside it. The flow asks for its allocations itself,
renews them and ends them, within `max_node_hours` (48 unless stated). It submits no job for
its own controller, which runs where the command is started. A site that keeps its accounts,
image and mounts in a file passes it to `intake derive --site`; that file is the site's, and
this tree names no account, partition, host or storage path. The block's keys are in
[task.example.yaml](task.example.yaml); what the pool does with them is
[DESIGN.md](DESIGN.md), "Compute".

## Through your agent

Open Claude Code or Codex in the TensorRT-LLM checkout and describe the task. The agent writes
the spec by following [spec/INTAKE.md](spec/INTAKE.md) and launches the run; what it cannot
find out it asks, and never guesses.

```text
Use the trtllm_modeling_v2 agent flow to onboard <model> from <checkpoint path>.
- Platform: <GPU type and count>; layouts: <TP / TEP / attention-DP sizes>.
- Workload: <input/output tokens>; KV cache: <formats>; features: <e.g. chunked prefill,
  block reuse, disaggregation, images>; exclude: <e.g. speculative decoding>.
- Accuracy: <benchmark and where its reference score comes from>. Report what is unsupported
  or not verified.
```

## While it runs

The session that started the run watches it with `monitor wait` and leaves the work to it:
never cancel a job the run submitted, and never run a replacement check beside it. A task that
cannot go on is `blocked`, and its receipt is its question; everything that does not depend on
it goes on, and one `resolve` verb answers it. [OPERATING.md](OPERATING.md) is the whole
protocol: the events, the verbs, and how to put one question to a person.

| A run ends | Meaning |
|---|---|
| `delivered` | every owed check passed on the delivered tree |
| `partial` | every task ended, and a task was skipped or a check was not verified: each is listed |
| exit 76 | everything left waits for an answer: `resolve`, then the same command |
| exit 75 | the controller stopped for the limit of the job it runs in: the same command goes on |

`onboarded` is pinned when the first deployment's session passes. The closing report
(`<workspace>/delivery_report.md`) says what each requested check found.

## Reference

[DESIGN.md](DESIGN.md): the terms, a run step by step, compute and trees, how reference code is
found, what a session is, the code map. A first run needs none of it.
