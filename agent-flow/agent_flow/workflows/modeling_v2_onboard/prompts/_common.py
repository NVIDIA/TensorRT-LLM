# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prose blocks shared by the planner and worker prompts.

The ModelingV2 doctrine here is adapted from the staircase-onboard role
prompts. It points at the package's own ``README.md`` and ``catalog/_op.py``
for anything they state, so the prompts do not restate a contract that the
code owns.
"""

from __future__ import annotations

import re

__all__ = [
    "CITATION_DISCIPLINE",
    "DETACHED_JOBS",
    "EVIDENCE_DISCIPLINE",
    "GPU_POOL",
    "MODELING_V2_LAYOUT",
    "MODELING_V2_SWITCH",
    "SLURM_ENVIRONMENT",
    "SOURCE_BOUNDARY",
    "tidy",
]


def tidy(text: str) -> str:
    """Collapse the blank-line runs that composing blocks leaves behind."""
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


MODELING_V2_LAYOUT = """\
## ModelingV2: where things live

ModelingV2 is one self-contained modeling codebase per deployment target
(checkpoint, GPU architecture, parallel topology), beside the built-in model
zoo. **Read `tensorrt_llm/_torch/_experimental/modeling_v2/README.md` first** —
it is the authoritative description of routing, layout, entries and gates.

| what | path |
|---|---|
| package README | `tensorrt_llm/_torch/_experimental/modeling_v2/README.md` |
| the entry contract (`OpWrapper`, `Cell`, `Arch`) | `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/_op.py` |
| shared entries | `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.py` |
| their GPU tests | `tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py` |
| a target | `tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/{modeling,weights}.py` |
| a target's own entries | `.../models/<family>/<ckpt>__<arch>__<parallel>/ops/<category>/<name>.py` |
| their GPU tests | `tests/unittest/_torch/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/<category>/test_modeling_v2_<name>.py` |
| routing | `tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/routing.py`, `_router_index.py` |
| source-level gates (no GPU) | `tests/unittest/_torch/modeling_v2/test_modeling_v2_*.py` |
| accuracy anchors | `tests/integration/defs/accuracy/references/<benchmark>.yaml` |
| accuracy gates | `tests/integration/defs/accuracy/test_modeling_v2_<family>.py` |

The existing targets (`models/gpt_oss/`, `models/deepseek_v3/`) are
gate-verified exemplars of every pattern. **Copy patterns, never values**:
every model-specific value is re-derived from this checkpoint.

- An entry is one `OpWrapper` subclass: one op call, a plain-torch
  `reference`, `is_valid`, the certified `CELLS` and `ARCHS`, and a `note`.
  It is stateless; `__call__` reads nothing off `self`.
- An entry only one target calls lives in that target's `ops/`; one that two
  or more targets call lives in `catalog/`. A target imports the catalog, the
  framework and its own `ops/` — never another target's code.
- Every test file is named `test_modeling_v2_<name>.py`.
"""

MODELING_V2_SWITCH = """\
## The modeling_v2 switch

`modeling_v2` is an LLM API argument: `off` (default), `auto` (use a target
when one matches, silently fall back otherwise), `require` (raise instead of
falling back). Pass it as `LLM(model, modeling_v2="require")` or
`modeling_v2: require` in a `trtllm-serve`/`trtllm-eval` config YAML.

**Anything you attribute to modeling_v2 runs under `require`.** Under `auto`
a configuration that misses the target's routing criteria silently gets the
built-in implementation, and a result measured that way reads as
modeling_v2's. It is the single most expensive mistake available here.

The built-in implementation under `modeling_v2="off"` is the natural
accuracy oracle when the engine already supports this architecture: same
harness, same workload, one variable.
"""

SOURCE_BOUNDARY = """\
## Reading the engine

TensorRT-LLM's source, including `_torch/models/modeling_<x>.py`, is an
ordinary read. **Read it to understand an interface — what an op expects,
what the runtime allocates, which metadata field an attention variant reads —
never to copy a structure.** A target's forward is one flat function built
from entries; the zoo's forward is a module tree built from shared helpers.
Code that starts mirroring an upstream module's shape is the signal that
something has gone wrong. A missing op is a missing entry (a `new_op` task),
never a reason to reach into the zoo for an implementation.

The HuggingFace reference modeling code is the ground truth of the **math**
(the checkpoint was trained against it); its op-by-op torch shape is never
the shape of target code. Other inference engines' modeling code stays out.
"""

EVIDENCE_DISCIPLINE = """\
## Evidence discipline

What you write in your result file is a **claim** that the next agent acts
on without re-checking.

- Every number is verbatim from tool output. If you did not measure it, say
  so.
- Re-check every claim against disk before you write it: a file that exists,
  a test that passed, a commit that landed.
- Separate what you observed from what you inferred.
- Report what you could not do. A blocked step named precisely is worth more
  than a step quietly skipped.
- Same-protocol accuracy reruns move by 0.1-0.3 points; a sub-0.5-point
  change is not a result.
"""

CITATION_DISCIPLINE = """\
## Writing down what you read

Name the file and a symbol the reader can grep (a function, class, field,
config key or exact assert text) — **never a line number**. Line numbers go
stale within days and read exactly like a true citation. A quoted diagnostic
(an error message with its own file:line) is the one exception: quote it
whole.
"""

DETACHED_JOBS = """\
## Long-running commands — launch them detached

Your session can be stopped and restarted (the orchestrator itself is
restarted). A blocking `srun`, long build or long eval dies with the session.
For anything longer than a few minutes, launch it detached through the
framework wrapper, which records it in your task's `jobs.json` so it survives
a restart:

    python -m agent_flow.jobs.run_detached \\
      --registry <your jobs.json> \\
      --handle <stable logical key, e.g. gsm8k-tp4> \\
      --kind slurm --name <task id>-<handle> \\
      -- <your sbatch script and args>

Use `--kind local` with the plain command when there is no Slurm. Use a
**stable** `--handle` that names what the job computes (no timestamps).

**Re-attach before you submit.** Read your `jobs.json` first. If a job is
already recorded under the handle, check it yourself (`squeue -j <id>`, then
`sacct -j <id> -o State,ExitCode`): still running → keep polling it; finished
→ collect its output; gone or failed → resubmit with the same `--handle`.
Never resubmit a live job.
"""

GPU_POOL = """\
## Running GPU work: the run's GPU pool

The run holds a pool of GPU nodes for its workers. Run every command that
needs a GPU through it with the command your turn prompt names; never submit
your own `srun` or `sbatch`. It waits until GPUs are free for you — work on
the way to the first accuracy result is served first — then runs the command
as a step of a pool job: inside the run's container, in your worktree
(mounted at its own path, as are the workspace and the checkout), with
`CUDA_VISIBLE_DEVICES` set to your GPUs. It exits with the command's status
and frees the GPUs when the command ends.

- `--gpus` is how many GPUs the command uses, `--minutes` how long it may
  run; ask for no more than it needs.
- Run anything longer than a few minutes detached with `--kind local` (see
  below), so it survives a restart of your session.
- Never run TensorRT-LLM code on the login node: only a pool command is
  evidence.

A fresh worktree has no native build artifacts. When `implements_tips` or
`test_commands.md` give a way to reuse an existing build, use it; the pool may
already do it for every command. `test_commands.md` in the workspace is a
cache of commands other tasks have **verified** to work: read it before
drafting a command, add or update an entry after one works (a one-line
purpose, then the exact command), and delete entries you find broken. Other
tasks edit it concurrently, so edit it in place with small changes.
"""

SLURM_ENVIRONMENT = """\
## Running GPU work on Slurm

`task.yaml` contains a `slurm-environment` section. Use its `slurm_partition`
and `docker_image` verbatim; do not invent another partition or image, and do
not fall back to a local run. If a value is unusable, report it as a blocker.

Every command that runs TensorRT-LLM code runs inside that container, with
the code **you are testing** mounted at its own absolute path:

- mount your task's worktree (it is a git worktree of `trtllm_repo_path`, so
  also mount `trtllm_repo_path` itself — the worktree's `.git` points into
  it) and the run's workspace, each at the same path inside the container;
- `cd` into your worktree inside the container before running anything.

A fresh worktree has no native build artifacts. When `implements_tips` or
`test_commands.md` give a way to reuse an existing build (for example linking
a built checkout's compiled artifacts into your worktree), use it: it takes
seconds, and is valid as long as your task changes no C++ or CUDA source.
Otherwise each fresh container session bootstraps the checkout it runs from:

```bash
cd <your worktree>
./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --skip_building_wheel -f --nvtx --use_ccache
pip install -e .[devel]
```

Run the bootstrap in the same `srun`/`sbatch` invocation as the command it
serves (`srun --partition=<slurm_partition> --container-image=<docker_image>
--container-mounts=<mounts> ...`). A TensorRT-LLM command on the login node,
or in a container that did not mount the code under test, is not evidence.

`test_commands.md` in the workspace is a cache of Slurm commands other tasks
have **verified** to work. Read it before drafting a command; after a command
works, add or update its entry (a one-line purpose, then the exact command).
Delete entries you find broken. Other tasks edit it concurrently, so edit it
in place with small changes rather than rewriting the file.
"""
