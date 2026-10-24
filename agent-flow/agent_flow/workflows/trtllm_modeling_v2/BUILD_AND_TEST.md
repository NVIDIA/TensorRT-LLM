# Build, rebuild, and test at each scale

A companion to [TROUBLESHOOTING.md](TROUBLESHOOTING.md). That file is keyed on
symptoms — *something broke, what is it*. This one is keyed on tasks — *I need
to build / rebuild / run something at N GPUs, how*.

Most of it is not this workflow's knowledge. It is collected from the places
in TensorRT-LLM that already do these things correctly, each named so you can
read the current version rather than trust this page:

| where | what it owns |
|---|---|
| `scripts/build_wheel.py` | the build itself, and every flag below |
| `.claude/skills/exec-slurm-compile/` | building on a Slurm cluster, and `enroot import` |
| `.claude/skills/trtllm-test-script-builder/` | generating Slurm test scripts — **the launch rules** |
| `.claude/skills/exec-local-slurm/` | persistent allocations, container reuse, step cancellation |
| `tests/integration/test_lists/test-db/*.yml` | which tests run at which GPU count |
| `scripts/test_to_stage_mapping.py` | which CI stage a given test lands in |

---

## 0. First ask: do you need to build at all?

**For a modeling_v2 onboard, almost certainly not.** This is worth checking
before spending an hour, because the answer is usually no:

| what you changed | rebuild? |
|---|---|
| a target's `modeling.py` / `weights.py` | **no** — pure Python, editable install picks it up |
| a catalog entry `.py`, shared or in a target's `ops/` | **no** |
| a FlashInfer call (sourcing tier 3) | **no** — the package is installed, its kernels JIT |
| a Triton kernel (sourcing tier 5) | **no** — Triton is JIT |
| routing, `_router_index.py` | **no** |
| anything under `cpp/` | yes |
| a new `torch.ops.trtllm.*` op | yes |
| pulling a trunk that touched `cpp/` | yes |

The five sourcing tiers this workflow uses — an exported op, the kernel plane,
FlashInfer's public API, native torch, Triton — **none of them require a native
rebuild**. That is why
the orchestrator never builds anything and `environment.python` just points at
an install someone else made.

If you are rebuilding during an onboard, stop and confirm why.

### Rebuilding while compute sessions are running

`build_wheel.py` installs `tensorrt_llm/libs/*.so` by rewriting each file in
place. Any process, on any node, that has the old file mapped then faults on
its next page-in, and when that process is attached to its allocation's MPI it
takes the whole allocation with it. Measured: a rebuild on one session killed
another session thirty seconds later.

So rebuild through the workflow's `incremental_build.sh` (in the package's
`scripts/` directory), as a whole-node spool job:

```bash
spoolctl.py submit --gpus all --wall 3h build-<what> \
    'bash <package>/scripts/incremental_build.sh --repo <checkout> --spool <spool A> --spool <spool B>'
```

It takes a per-checkout build lock, announces the rebuild on every listed
spool (each session's scheduler then holds new GPU jobs against that checkout),
gives every installed library a fresh inode by copy-and-rename so running
processes keep the file they mapped, runs the build with ccache, checks the
checkout still imports, and retracts the announcement — also when the build
fails. Spools default to `$MV2FLOW_SESSION_SPOOLS`, which every rendered
session exports.

## 1. Build

```bash
python3 ./scripts/build_wheel.py \
    --use_ccache \
    --skip_building_wheel \
    --cuda_architectures "100-real;103-real"
```

| flag | when |
|---|---|
| `--cuda_architectures` | **always set it.** `100-real` Blackwell B200/GB200, `103-real` GB300, `90-real` Hopper, `89-real` L40S, `80-real` A100. Building all of them is the slowest thing you can do |
| `--use_ccache` | always. A no-op change rebuilds in minutes instead of an hour |
| `--skip_building_wheel` | development — builds in place, no wheel artifact |
| `--fast_build` / `-f` | skips some kernels. Development only — **do not gate on a fast build** |
| `--clean` / `-c` | wipes the build directory. The answer to "weird compile issues" |
| `--configure_cmake` | force a CMake re-configure. Needed when a CMake input changed but the build did not notice |
| `-j <N>` | lower it when the build OOMs |

`build_wheel.py` creates its own venv and runs `pip install -e .[devel]` into
it, so the interpreter it produces is the one to put in `environment.python`.

### ccache does not carry across checkouts

Measured: pointing `CCACHE_DIR` at another checkout's cache produced no
useful hit rate. Absolute paths are baked into the cache keys. Give each
checkout its own, and expect the first build there to be cold.

### The one that wastes an afternoon

A rebuild is *not* the fix for `NVRTC ... could not open source file "cuda.h"`.
Measured: rebuilding with `--nvrtc_dynamic_linking` (98% ccache hits) did not
fix it; the cause was the venv's `bin` missing from `PATH`. See
TROUBLESHOOTING.md before you rebuild for that symptom.

## 2. Build on Slurm

`.claude/skills/exec-slurm-compile/` owns this end to end — `submit_compile.sh`
submits, `compile.slurm` allocates, `compile.sh` runs the build inside the
container, `enroot-import` fetches the image.

Two failure modes from its own troubleshooting table that are not obvious:

- `enroot import` **auth** failure → `~/.config/enroot/.credentials`;
- `enroot import` producing an **empty or corrupt** `.sqsh` → the image URL
  probably has an `https://` prefix it should not have.

### The environment hazard on a shared cluster

`/root/.local` and `/root/.cache` may be symlinked into a shared home, so
another checkout's develop-install is on the path of every container run.

- **At runtime: `PYTHONNOUSERSITE=1`.** Without it an import can resolve to a
  different tree than the one you are measuring.
- **During the native build: leave the user site enabled.** The CUTLASS kernel
  CMake runs `setup_library.py develop --user` and then imports
  `cutlass_library`. Isolate the `pip install -e` step and everything after
  it, not the compile itself.

## 3. Test — the scale ladder

Scale is declared by `system_gpu_count` in each list's `condition` block, and
the file name follows it:

| scale | list | `system_gpu_count` | what belongs here |
|---|---|---|---|
| **CPU** | `l0_cpu.yml` | `0–0` | anything that needs no device. `backend: generic` |
| **1–4 GPU** | `l0_gb300.yml`, `l0_b200.yml`, `l0_h100.yml`, … | `1–4` | the ordinary pre-merge matrix |
| **multi-GPU** | `l0_*_multi_gpus.yml` | `4–4`, `8–8` | collectives, TP/EP topologies |
| **multi-node** | `l0_*_multi_nodes*.yml` | `8–8`+ | anything spanning nodes |

To find where a test you care about runs:

```bash
python scripts/test_to_stage_mapping.py --tests "test_modeling_v2_allgather"
```

### CPU

`agent-flow`'s own suite is the clearest example: pure CPU, its own stage
(`CPU-AgentFlow-UnitTest`), and `AgentFlowRule` narrows CI so an
agent-flow-only change runs exactly that stage and no GPU stage at all.

For the engine, CPU tests are the schema / list / prose audits — including the
modeling_v2 claims audits, which are text scans and need no device.

### Single GPU

```bash
pytest tests/unittest/_torch/modeling_v2/norm/test_modeling_v2_flashinfer_rmsnorm.py
LLM_MODELS_ROOT=/path/to/models pytest tests/integration/defs/accuracy/test_modeling_v2_gpt_oss.py
```

Unit tests need no models. Integration tests need `LLM_MODELS_ROOT` **and** a
GPU; forgetting the variable produces a failure that reads like a missing
model rather than a missing setting.

### Multi-GPU: two mechanisms, and they are not interchangeable

**`mpi_pool_executor`** — a fixture, worker count by indirect parametrization:

```python
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_something(mpi_pool_executor): ...
```

**A launcher in a fresh interpreter** — what the modeling_v2 collective
entries use, and the reason is stated in
`tests/unittest/_torch/modeling_v2/comm/_rank_job.py`:

> A collective that breaks \*hangs\* rather than raising, so the
> deadline is load-bearing and the ``mpi_pool_executor`` fixture has none.

Two consequences worth carrying into any new multi-rank test:

- **A hang is the normal failure.** Whatever you write needs a deadline that
  kills the process group, or one wedged rank takes the whole run with it.
- **You cannot start an MPI launcher from a pytest process that has already
  imported `tensorrt_llm`** — that process has initialized MPI. The launcher
  must go in a fresh interpreter, started *by file path*.

### Multi-node

The launch rules are the whole content here, and they come from
`.claude/skills/trtllm-test-script-builder/`. **The rule is a function of the
task count and nothing else:**

| | `ntasks == 1` | `ntasks > 1` |
|---|---|---|
| `--mpi=` | `none` | `pmix` (or the cluster's variant — ask `srun --mpi=list`) |
| payload | clear the MPI environment, then exec | wrap in `trtllm-llmapi-launch` |

```bash
srun -N 2 --ntasks 8 --ntasks-per-node 4 --mpi=pmix --gres=gpu:4 \
    --container-image=... --container-name=trtllm-run \
    bash -c "trtllm-llmapi-launch python3 -m tensorrt_llm.commands.serve $MODEL ..."
```

Four things that are each a whole afternoon if you get them wrong:

1. **`--ntasks` must match the world size.** Without it `srun` launches one
   task, the MPI world size is 1, and the LLM API concludes it must create the
   other ranks itself. `need_spawn_mpi_workers` in `llmapi/mpi_session.py` is
   exactly `get_mpi_world_size() == 1 and model_world_size > 1`. Under pyxis
   that dynamic spawn is forbidden → `MPI_ERR_SPAWN`.
2. **`--gres` is per node**, not per world. They diverge the moment you span
   nodes.
3. **At one task, clear the MPI environment.** mpi4py auto-initializes MPI
   whenever PMI variables are present, and inside a PMIx-managed step they
   always are — so a single-task job initializes MPI for no reason and then
   dies spawning a worker. Keep `SLURM_PROCID` and `SLURM_LOCALID`; they
   identify the rank rather than bootstrap MPI.
4. **`TLLM_WORKER_USE_SINGLE_PROCESS=1` is not the general fix.** The function
   behind it is named `enable_worker_single_process_for_tp1` and the source
   calls it a WAR for TP1. It does not scale past one rank.

Inside a multi-node compute session there is no `srun` and no sshd in the
container. A session declared with `nodes: 2` runs one scheduler per node, and
the workflow's `mn_mpirun.sh` (in the package's scripts directory, under
`mn/`, with its own README) starts one `mpirun` across them, launching each remote daemon through
that node's scheduler.

Inside a spool session the workflow emits the `mpirun` form of the same rule
from `environment.py` (`mpirun -n <world> ... trtllm-llmapi-launch`); nothing
here is an agent's decision. The `srun` form is written out for a human
launching a test by hand, who needs to recognize a correct line.

## 4. Iterating without paying full price

| cost | how to stop paying it |
|---|---|
| container import, 26 GB / 4–8 min **per job** | `--container-name=<name>` — pyxis skips the import when that name already exists on the node |
| queue wait per command | a persistent allocation: `salloc` once, then `srun --jobid=<id>` steps into it (see `exec-local-slurm`) |
| both, plus a transport that cannot reach Slurm at all | a compute session — one long-lived job whose **own main process** is the spool scheduler; work is a file |
| cold native build | `--use_ccache`, and one cache per checkout |

On the spool: work started inside a `slurm_exec_in_job` step does **not**
survive the step's teardown, `nohup` included — measured. The long-running
process has to be the job's own main process. If your cluster's Slurm CLI is
reachable, prefer the persistent-allocation pattern; the spool exists for the
case where it is not.

A session is declared in `task.yaml` (`environment.sessions`) and its entry
script is rendered, never written by hand:

```bash
trtllm-modeling-v2 session render --task task.yaml --name a      # <spool>/session_entry.sh
trtllm-modeling-v2 session submission --task task.yaml --name a  # the broker call that starts it
trtllm-modeling-v2 session status --task task.yaml               # heartbeat, end, queue
```

Inside a session, size each job instead of picking devices:

| you run | submit it as |
|---|---|
| a single-GPU unit test or microbenchmark | `--gpus 1` |
| a 4-rank collective test | `--gpus 4` |
| anything that loads the full model, a gate, a rebuild | `--gpus all` |
| a worktree's code | add `--tree <worktree>` — sets the working directory **and** `PYTHONPATH`, and refuses a result whose ranks imported another checkout |
| the orchestrator, or anything GPU-free and long-lived | a `free-` name, no `--gpus` |

`--wall` counts from the job's start, so time queued behind other work is
free. `spoolctl.py status` says why each waiting job waits; `spoolctl.py cancel
<name>` is the only way to stop one (see the safe-kill rule in
TROUBLESHOOTING.md).

To abort one run without losing the allocation:

```bash
squeue -s -j <job_id>      # list steps
scancel <job_id>.0         # cancel one step
```

## 5. Where a new test belongs

1. **No device?** → a CPU list, and check whether a CBTS rule already narrows
   CI for your subtree (`jenkins/scripts/cbts/rules/`).
2. **One device?** → the ordinary `l0_<gpu>.yml` for the architecture you
   certified on. Certification is per-architecture (`ARCHS`); a pass elsewhere
   is not a pass here.
3. **Collectives or a TP/EP topology?** → `l0_*_multi_gpus.yml`, and give it a
   deadline.
4. **Spans nodes?** → `l0_*_multi_nodes*.yml`, and it needs the launch rules
   in §3.

Then confirm with `scripts/test_to_stage_mapping.py` that it actually lands in
a stage. A test on no list is a test nobody runs.

**One trap for documentation-shaped changes:** `ModelingV2Rule` deliberately
excludes `*.md` so a documentation-only edit does not pull in multi-GPU GB300
stages. That is the right trade, but it means a check that *scans* `.md` files
gets no CI trigger from the PRs most likely to break it. Put text audits in a
pre-commit hook, not in a GPU test stage.
