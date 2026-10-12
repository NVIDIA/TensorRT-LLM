# F006 — two multi-rank launch defects preflight cannot see

- **Found:** 2026-09-22 05:34-05:37 PDT, running the substrate gate by hand
  through the compute session before committing the run
- **Component:** `agent-flow/agent_flow/workflows/staircase_onboard/environment.py`
  (`command`, `_local_prefix`) and `preflight.py` (`check_tmpdir`)
- **Severity:** every `session_spool` / `local` run at world size > 1 fails,
  twice, with errors that name neither the spec nor the workflow
- **Status:** both reproduced and both fixed by hand for this run; fixes proposed

Both of these are exactly the class `TROUBLESHOOTING.md` exists for: the error
text mentions Open MPI and ZeroMQ, and the cause is a line in `task.yaml`.

---

## Defect 1 — `mpirun` gets one slot, because the session allocation has one task

`RunEnvironment.command` wraps a world-size-4 payload as:

```
mpirun -n 4 --allow-run-as-root trtllm-llmapi-launch <python> ...
```

Correct in intent — the ranks must exist before the engine starts, or the LLM
API concludes it has to spawn them and dies with `MPI_ERR_SPAWN`. But inside a
Slurm allocation Open MPI takes its slot count from the resource manager, and
a spool session is submitted as **one task** that owns the node:

```
NumNodes=1 NumCPUs=144 NumTasks=1 ... TresPerNode=gres/gpu:4
```

So `-n 4` asks for four slots against one, and the run dies before any Python:

```
There are not enough slots available in the system to satisfy the 4
slots that were requested by the application:
  trtllm-llmapi-launch
```

**Why the obvious fix is wrong.** Submitting the session with
`ntasks_per_node=4` gives MPI its four slots and simultaneously starts **four
copies of the spool watcher**, which then race for every piece of work. The
session is one process by design.

**Fix: `--oversubscribe`.** The allocation owns the whole node — 144 CPUs and
all 4 GPUs — so the slot count is a bookkeeping guard, not a real constraint.
Verified: with `--oversubscribe` the four ranks start and bind
(`rank 0 using MpiCommSession to bind to external MPI processes`, four times).

It belongs in `environment.py`'s non-Slurm branch, next to the existing
`mpirun` assembly, with a comment saying why — the `kind: slurm` branch must
**not** get it, because there `srun --ntasks` really does size the world and
oversubscribing would hide a mis-sized allocation.

---

## Defect 2 — a long `TMPDIR` breaks the LLM API's RPC, at 107 characters

With the slots fixed, all four ranks started and then failed:

```
zmq.error.ZMQError: ipc path
"/scratch/<fs>/portfolios/<org>/projects/<project>/users/<user>/staircase-tmp/rpc_test_9bbff7c6-e827-43d1-9287-d9627067e179"
is longer than 107 characters (sizeof(sockaddr_un.sun_path)).
```

and then, from the same path, `zmq.error.ZMQError: File name too long`.

107 is `sizeof(sockaddr_un.sun_path)` — a kernel limit on unix domain socket
paths, not a ZeroMQ or TensorRT-LLM choice. The LLM API's RPC binds an IPC
socket under `TMPDIR` named `rpc_test_<uuid4>`, which is a fixed 45-character
suffix. So **any `TMPDIR` longer than 62 characters fails**, and this run's
was 92.

`check_tmpdir` asserts `TMPDIR` is set and not under `/tmp`, for a good and
well-documented reason. It says nothing about length, and length is the
property that breaks first on a cluster whose storage paths are deep —
`/scratch/<fs>/portfolios/<org>/projects/<project>/users/<user>/`
was 78 characters on that cluster before the run added anything.

**Fix: assert the length in `check_tmpdir`,** with the arithmetic in the
message:

> `TMPDIR=<value>` is 92 characters. The LLM API binds an RPC socket at
> `$TMPDIR/rpc_test_<uuid4>`, and a unix socket path is capped at 107
> (`sizeof(sockaddr_un.sun_path)`), so `TMPDIR` must be at most 62. Point it
> at a shorter path on the same storage.

Seconds in preflight instead of a four-rank engine load and a ZeroMQ traceback.

**What this run did.** `/lustre/<fs>/portfolios/<org>/users/<user>/tmp4`,
49 characters — the *same storage* as the 92-character `/scratch` spelling
(established earlier in this run), just a shorter path to it. Measured in the
session: a `rpc_test_<uuid4>` path under it is 95 characters and binds.

**A related fact worth recording, because it contradicts an earlier note.**
Unix sockets bind fine on this Lustre filesystem — tested directly on
`/tmp`, `/dev/shm` and both Lustre spellings, all four succeed. The earlier
guidance that a Lustre `TMPDIR` breaks torch's shared-memory manager is
therefore either about a different mount or about a different limit; on this
cluster the failure is **path length**, not filesystem type. Anyone copying
the `TMPDIR: /tmp` workaround from the multimodal options file should know
that it works because `/tmp` is *short*, not because it is local.

---

## Why neither was caught earlier

`--dry-run` skips the substrate gate's actual execution under `session_spool`
— the devices are the scheduler's, preflight claims none, and the gate is the
first thing that would have run four ranks. Both defects live in the gap
between "the spec validates" and "the first real command runs".

That gap is what the substrate gate is for, and it would have caught both on
the real run. The cost would have been one failed preflight per defect rather
than two hand-run probes — which is the right outcome and an argument for
running the substrate gate rather than reaching for `--skip-substrate-gate`.
