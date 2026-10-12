# F007 — Slurm's env defeats `trtllm-llmapi-launch` under `mpirun`, silently

- **Found:** 2026-09-22 05:55-06:05 PDT, running the substrate gate by hand
- **Component:** `agent-flow/.../staircase_onboard/environment.py`, `RunEnvironment.command`
  — and the underlying cause is in TensorRT-LLM's own `trtllm-llmapi-launch`
- **Severity:** every non-`slurm` launch at world size > 1 inside a Slurm
  allocation hangs forever, having silently become N independent single-rank runs
- **Status:** root-caused, proven both ways, fixed by hand for this run;
  workflow fix proposed and an upstream note recommended

## The symptom

The substrate gate started, printed its banners, and then stopped. The log did
not grow for ten minutes. No error, no traceback, no exit.

`AGENTS.md` lists this exact shape:

> the engine hangs at startup and never returns — a wrong launch substrate
> hangs rather than raises

## What was actually happening

A process listing on the node:

```
178503  python3 -m tensorrt_llm.llmapi.mgmn_leader_node
178507  python3 -m tensorrt_llm.llmapi.mgmn_leader_node
178510  python3 -m tensorrt_llm.llmapi.mgmn_leader_node
178511  python3 -m tensorrt_llm.llmapi.mgmn_leader_node
```

**Four leaders.** A correct 4-rank launch has one leader and three workers.
GPU 0 held 3 GB and GPUs 1-3 were empty, so nothing was computing — four
processes were each waiting to be joined by workers that would never come.

## Root cause

`trtllm-llmapi-launch` derives its rank and world size like this:

```sh
mpi_rank=${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMI_ID:-0}}}}
```

and, in `mpi_world_size`, checks `SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE`.

**Slurm wins over Open MPI in both.** That is the right precedence under
`srun`, which is what the launcher was written for. It is exactly wrong under
`mpirun` *inside* a Slurm allocation — and a `session_spool` job is precisely
that: one Slurm task that owns the node, running `mpirun` itself.

Measured in the session, four ranks, one variable changed:

| | `SLURM_PROCID` | `OMPI_COMM_WORLD_RANK` | `SLURM_NTASKS` | `OMPI_COMM_WORLD_SIZE` |
|---|---|---|---|---|
| as-is | `0, 0, 0, 0` | `0, 1, 2, 3` | `1` | `4` |
| SLURM vars scrubbed | unset | `0, 1, 2, 3` | unset | `4` |

So every rank read `mpi_rank=0` and became a leader, and each also read
`mpi_world_size=1` — the launcher was wrong about *both* numbers, from the
same cause.

MPI itself was never at fault. A bare `mpi4py` probe under the identical
`mpirun` line reported `rank 0..3 of 4` correctly.

## The fix

Scrub the Slurm task variables immediately before `mpirun`, so the launcher
falls through to the Open MPI ones it should be reading:

```
env -u SLURM_PROCID -u SLURM_NTASKS -u SLURM_LOCALID -u SLURM_NODEID \
    -u SLURM_STEP_ID -u SLURM_TASK_PID -u SLURM_STEP_NUM_TASKS \
    mpirun -n 4 --oversubscribe --allow-run-as-root trtllm-llmapi-launch ...
```

Verified end to end on the substrate gate, stock path, 4x GB300:

```
mpi_rank: 0 / 1 / 2 / 3
Rank0 run mgmn leader node with mpi_world_size: 4
Rank1/2/3 run mgmn worker node with mpi_world_size: 4
[0] Prompt: 'The capital of France is'..., Generated text: ' Paris.\nThe'...
Rank0 Task exit code: 0
```

## Where it belongs in the workflow

`environment.py` already owns the inverse of this rule. `SINGLE_TASK_MPI_SCRUB`
exists because, at world size 1, "mpi4py auto-initializes MPI whenever PMI
variables are present and inside a PMIx step they always are". That is the
same insight — a Slurm step's environment misleads a process that is not being
launched by `srun` — applied to the other end of the range.

So the change is to extend the existing rule rather than invent one:

- the **`slurm`** branch keeps today's behaviour exactly. There `srun --ntasks`
  really does size the world, `SLURM_PROCID` really is the rank, and scrubbing
  would break a correct launch.
- the **`mpirun`** branch (`local`, `session_spool`) gains the scrub, for the
  same reason the world-size-1 branch has one.

Together with F006's `--oversubscribe`, the non-Slurm multi-rank line becomes
complete. Both are properties of "running `mpirun` inside someone else's
allocation", and neither is a property of this model, this target or this run.

## Worth reporting upstream

`trtllm-llmapi-launch` preferring `SLURM_PROCID` unconditionally is defensible
but produces the worst available failure mode when the assumption breaks: not
an error, not a wrong answer, but N silent single-rank runs that hang forever.
A cheap guard would be to compare the two when both are present and refuse to
start when they disagree — `SLURM_PROCID=0, OMPI_COMM_WORLD_RANK=3` is not a
situation any correct launch produces.

The same applies to `SLURM_NTASKS=1` alongside `OMPI_COMM_WORLD_SIZE=4`.

## Note on the UCX errors, which are NOT this

Throughout, every rank logged:

```
UCX ERROR uct_iface_open(tcp/rdma_vf_rail0) failed: Input/output error
pml_ucx.c:314  Error: Failed to create UCP worker
```

These are noise here and were a red herring for a while. UCX fails to open one
IPv6 RDMA virtual-function interface and falls back; the run completes with
exit code 0 and correct output. `UCX_TLS=sm,self` silences them and is worth
setting to keep the logs readable, but it fixes nothing — confirmed by the
passing run above, which still logs them.
