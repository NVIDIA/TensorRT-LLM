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
restarted). A command that blocks your session dies with it. Launch anything
longer than a few minutes detached through `agent_flow.jobs.run_detached`
(its `--help` gives the usage). It records the job in your task's `jobs.json`
so a restarted session finds it. Give each job a **stable** handle that names
what it computes (no timestamps).

`jobs.json` belongs to the wrapper: never write or edit it yourself. Keep
your own notes about jobs in another file of your node directory.

**Re-attach before you launch.** Read your `jobs.json` first. A job already
recorded under the handle that is still running is polled, not relaunched; a
finished one is collected; only a gone or failed one is launched again under
the same handle. Check a recorded Slurm job yourself with `squeue -j <id>`,
then `sacct -j <id> -o State,ExitCode` once it has left the queue.
"""

GPU_POOL = """\
## Running GPU work: the run's GPU pool

The run holds a pool of GPU nodes for its workers. Every command that needs a
GPU goes through the pool, with the entry point your turn prompt names; never
submit your own Slurm jobs. The pool waits until GPUs are free for you (work
on the way to the first accuracy result is served first), runs the command in
the run's container and your worktree with only your GPUs visible, and frees
them when the command ends.

- Ask for no more GPUs and time than the command needs.
- A long pool command is launched detached (see below), so it survives a
  restart of your session.
- Never run TensorRT-LLM code on the login node: only a pool command is
  evidence.

A fresh worktree has no native build artifacts, and the container knows
nothing of the run's Python environment or model paths. The run's environment
task records the way to prepare both in `test_commands.md`; use it, and if it
is not recorded yet, write code and wait for it rather than building your
own. `test_commands.md` in the workspace is a cache of commands other tasks
have **verified** to work: read it before drafting a command, record one after
it works, and delete entries you find broken. Other tasks edit it
concurrently, so edit it in place with small changes.
"""

SLURM_ENVIRONMENT = """\
## Running GPU work on Slurm

`task.yaml` contains a `slurm-environment` section. Use its partition and
container image as given; do not invent others, and do not fall back to a
local run. If a value is unusable, report it as a blocker.

Every command that runs TensorRT-LLM code runs inside that container, from
the code **you are testing**: your task's worktree, the checkout it belongs
to and the run's workspace are mounted at their own paths, and the command
runs in your worktree. A TensorRT-LLM command on the login node, or in a
container without the code under test, is not evidence.

A fresh worktree has no native build artifacts. When `implements_tips` or
`test_commands.md` give a way to reuse an existing build, use it; it is valid
as long as your task changes no C++ or CUDA source. Otherwise each fresh
container session builds the checkout it runs from, in the same job as the
command it serves:

```bash
cd <your worktree>
./scripts/build_wheel.py --trt_root /usr/local/tensorrt --benchmarks --skip_building_wheel -f --nvtx --use_ccache
pip install -e .[devel]
```

`test_commands.md` in the workspace is a cache of commands other tasks have
**verified** to work: read it before drafting a command, record one after it
works, and delete entries you find broken. Other tasks edit it concurrently,
so edit it in place with small changes.
"""
