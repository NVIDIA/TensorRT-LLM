# Build, rebuild, and test at each scale

Keyed on tasks: *I need to build, rebuild or run something at N GPUs, how*.
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) is keyed on symptoms. Most of this page
points at the places in TensorRT-LLM that already do these things; read the
current version there rather than trust this page:

| where | what it owns |
|---|---|
| `scripts/build_wheel.py` | the build itself, and every flag below |
| `.claude/skills/exec-slurm-compile/` | building on a Slurm cluster, and `enroot import` |
| `.claude/skills/trtllm-test-script-builder/` | generating Slurm test scripts: **the launch rules** |
| `.claude/skills/exec-local-slurm/` | persistent allocations, container reuse, step cancellation |
| `tests/integration/test_lists/test-db/*.yml` | which tests run at which GPU count |
| `scripts/test_to_stage_mapping.py` | which CI stage a given test lands in |

---

## 0. First ask: do you need to build at all?

**For a modeling_v2 onboard, almost certainly not.**

| what you changed | rebuild? |
|---|---|
| a target's `modeling.py` / `weights.py` | **no**: pure Python, the editable install picks it up |
| a catalog entry `.py`, shared or in a target's `ops/` | **no** |
| a FlashInfer call (sourcing tier 3) | **no**: the package is installed, its kernels JIT |
| a Triton kernel (sourcing tier 5) | **no**: Triton is JIT |
| routing, `_router_index.py` | **no** |
| anything under `cpp/` | yes |
| a new `torch.ops.trtllm.*` op | yes |
| pulling a trunk that touched `cpp/` | yes |

None of the five sourcing tiers needs a native rebuild, so the orchestrator
never builds anything and `environment.python` points at an install someone
else made. If you are rebuilding during an onboard, confirm why.

### Rebuilding while compute sessions are running

`build_wheel.py` installs `tensorrt_llm/libs/*.so` by rewriting each file in
place: any process, on any node, that has the old file mapped faults on its
next page-in, and one attached to its allocation's MPI takes the whole
allocation with it (measured: a rebuild on one session killed another thirty
seconds later). Rebuild through the workflow's `incremental_build.sh` (in the
package's `infra/scripts/`), as a whole-node spool job:

```bash
spoolctl.py submit --gpus all --wall 3h build-<what> \
    'bash <package>/infra/scripts/incremental_build.sh --repo <checkout> --spool <spool A> --spool <spool B>'
```

It takes a per-checkout build lock, announces the rebuild on every listed spool
(each session's scheduler then holds new GPU jobs against that checkout), gives
every installed library a fresh inode by copy-and-rename so running processes
keep the file they mapped, builds with ccache, checks the checkout still
imports, and retracts the announcement, also when the build fails. Spools
default to `$MV2FLOW_SESSION_SPOOLS`, which every rendered session exports. The
script runs its helper (`infra/substrate/rebuild.py`) under `$MV2FLOW_PYTHON`,
else the `python3` on `PATH`, and refuses an interpreter older than 3.7.

## 1. Build

```bash
python3 ./scripts/build_wheel.py \
    --use_ccache \
    --skip_building_wheel \
    --cuda_architectures "100-real;103-real"
```

| flag | when |
|---|---|
| `--cuda_architectures` | **always.** `100-real` B200/GB200, `103-real` GB300, `90-real` Hopper, `89-real` L40S, `80-real` A100. Building all of them is the slowest thing you can do |
| `--use_ccache` | always: a no-op change rebuilds in minutes instead of an hour |
| `--skip_building_wheel` | development: builds in place, no wheel |
| `--fast_build` / `-f` | skips some kernels. Development only: **do not gate on a fast build** |
| `--clean` / `-c` | wipes the build directory |
| `--configure_cmake` | forces a CMake re-configure when a CMake input changed unnoticed |
| `-j <N>` | lower it when the build OOMs |

`build_wheel.py` creates its own venv and runs `pip install -e .[devel]` into
it; that interpreter is the one to put in `environment.python`. ccache does not
carry across checkouts (absolute paths are baked into the keys): give each
checkout its own cache, and expect its first build to be cold. A rebuild is
*not* the fix for `NVRTC ... could not open source file "cuda.h"`; see
TROUBLESHOOTING.md.

## 2. Build on Slurm

`.claude/skills/exec-slurm-compile/` owns this end to end: `submit_compile.sh`
submits, `compile.slurm` allocates, `compile.sh` builds inside the container,
`enroot-import` fetches the image. Two failure modes from its troubleshooting
table: an `enroot import` **auth** failure (`~/.config/enroot/.credentials`),
and an **empty or corrupt** `.sqsh` (an image URL with an `https://` prefix it
should not have).

On a shared cluster `/root/.local` and `/root/.cache` may be symlinked into a
shared home, so another checkout's develop-install is on the path of every
container run. Set **`PYTHONNOUSERSITE=1` at runtime**, but **leave the user
site enabled during the native build**: the CUTLASS kernel CMake runs
`setup_library.py develop --user` and then imports `cutlass_library`.

## 3. Test — the scale ladder

Scale is declared by `system_gpu_count` in each list's `condition` block, and
the file name follows it:

| scale | list | `system_gpu_count` | what belongs here |
|---|---|---|---|
| **CPU** | `l0_cpu.yml` | `0–0` | anything that needs no device. `backend: generic` |
| **1–4 GPU** | `l0_gb300.yml`, `l0_b200.yml`, `l0_h100.yml`, … | `1–4` | the ordinary pre-merge matrix |
| **multi-GPU** | `l0_*_multi_gpus.yml` | `4–4`, `8–8` | collectives, TP/EP topologies |
| **multi-node** | `l0_*_multi_nodes*.yml` | `8–8`+ | anything spanning nodes |

```bash
python scripts/test_to_stage_mapping.py --tests "test_modeling_v2_allgather"   # where a test runs
```

### CPU

`agent-flow`'s own suite is pure CPU, in its own stage
(`CPU-AgentFlow-UnitTest`), and `AgentFlowRule` narrows CI so an
agent-flow-only change runs that stage and no GPU stage. For the engine, CPU
tests are the schema, list and prose audits, including the modeling_v2 claims
audits.

### Single GPU

```bash
pytest tests/unittest/_torch/modeling_v2/norm/test_modeling_v2_flashinfer_rmsnorm.py
LLM_MODELS_ROOT=/path/to/models pytest tests/integration/defs/accuracy/test_modeling_v2_gpt_oss.py
```

Unit tests need no models. Integration tests need `LLM_MODELS_ROOT` **and** a
GPU; without the variable the failure reads like a missing model.

### Multi-GPU: two mechanisms, and they are not interchangeable

**`mpi_pool_executor`**, a fixture with the worker count by indirect
parametrization:

```python
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_something(mpi_pool_executor): ...
```

**A launcher in a fresh interpreter**, which the modeling_v2 collective entries
use (`tests/unittest/_torch/modeling_v2/comm/_rank_job.py`): a collective that
breaks hangs rather than raising, and the fixture has no deadline. So a new
multi-rank test needs a deadline that kills the process group, and an MPI
launcher cannot start from a pytest process that has already imported
`tensorrt_llm` (that process initialized MPI): start it in a fresh interpreter,
by file path.

### Multi-node

The launch rules come from `.claude/skills/trtllm-test-script-builder/`, and
they are a function of the task count alone:

| | `ntasks == 1` | `ntasks > 1` |
|---|---|---|
| `--mpi=` | `none` | `pmix` (or the cluster's variant: `srun --mpi=list`) |
| payload | clear the MPI environment, then exec | wrap in `trtllm-llmapi-launch` |

```bash
srun -N 2 --ntasks 8 --ntasks-per-node 4 --mpi=pmix --gres=gpu:4 \
    --container-image=... --container-name=trtllm-run \
    bash -c "trtllm-llmapi-launch python3 -m tensorrt_llm.commands.serve $MODEL ..."
```

1. **`--ntasks` must match the world size.** Without it `srun` launches one
   task, the MPI world size is 1, and the LLM API spawns the other ranks itself
   (`need_spawn_mpi_workers` in `llmapi/mpi_session.py`), which pyxis forbids:
   `MPI_ERR_SPAWN`.
2. **`--gres` is per node**, not per world.
3. **At one task, clear the MPI environment**: mpi4py initializes MPI whenever
   PMI variables are present. Keep `SLURM_PROCID` and `SLURM_LOCALID`, which
   identify the rank.
4. **`TLLM_WORKER_USE_SINGLE_PROCESS=1` is not the general fix**: it is a
   workaround for TP1 and does not scale past one rank.

Inside a spool session the workflow emits the `mpirun` form of the same rule
from `environment.py` (`mpirun -n <world> ... trtllm-llmapi-launch`); no agent
composes it. The container has no `srun` and no sshd: a session declared with
`nodes: 2` runs one scheduler per node, and `infra/scripts/mn/mn_mpirun.sh`
starts one `mpirun` across them ([infra/scripts/mn/README.md](infra/scripts/mn/README.md)).

## 4. Iterating without paying full price

| cost | how to stop paying it |
|---|---|
| container import, 26 GB / 4–8 min **per job** | `--container-name=<name>`: pyxis skips the import when that name already exists on the node |
| queue wait per command | a persistent allocation: `salloc` once, then `srun --jobid=<id>` steps into it (see `exec-local-slurm`) |
| both, plus a transport that cannot reach Slurm at all | a compute session: one long-lived job whose **own main process** is the spool scheduler; work is a file |
| cold native build | `--use_ccache`, and one cache per checkout |

Work started inside a `slurm_exec_in_job` step does **not** survive the step's
teardown, `nohup` included (measured): the long-running process has to be the
job's own main process. Where the Slurm CLI is reachable, prefer the
persistent-allocation pattern. A session is declared in `task.yaml`
(`environment.sessions`) and its entry script is rendered, never written by
hand:

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
| a worktree's code | add `--tree <worktree>`: it sets the working directory **and** `PYTHONPATH`, and refuses a result whose ranks imported another checkout |
| the orchestrator, or anything GPU-free and long-lived | a `free-` name, no `--gpus` |

`--wall` counts from the job's start. `spoolctl.py status` says why each
waiting job waits; `spoolctl.py cancel <name>` is the only way to stop one (the
safe-kill rule in TROUBLESHOOTING.md). To abort one run without losing a
persistent allocation: `squeue -s -j <job_id>` lists its steps, and
`scancel <job_id>.0` cancels one.

## 5. Where a new test belongs

1. **No device?** A CPU list; check whether a CBTS rule already narrows CI for
   your subtree (`jenkins/scripts/cbts/rules/`).
2. **One device?** The ordinary `l0_<gpu>.yml` for the architecture you
   certified on: certification is per-architecture (`ARCHS`).
3. **Collectives or a TP/EP topology?** `l0_*_multi_gpus.yml`, with a deadline.
4. **Spans nodes?** `l0_*_multi_nodes*.yml`, with the launch rules in §3.

Then confirm with `scripts/test_to_stage_mapping.py` that it lands in a stage:
a test on no list is a test nobody runs. `ModelingV2Rule` excludes `*.md` so a
documentation-only edit does not pull in multi-GPU stages, which means a check
that *scans* `.md` files gets no CI trigger from the PRs most likely to break
it: put text audits in a pre-commit hook, not in a GPU test stage.
