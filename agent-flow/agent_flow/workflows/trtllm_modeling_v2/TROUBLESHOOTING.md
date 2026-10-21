# Troubleshooting — infrastructure, not modeling

Everything here cost a real run real hours, and every one of them **read as a
model or target defect while it was happening**. That is the pattern worth
internalizing: an engine that will not boot, a gate that reports a 27-point
regression, a test that says nine skills are malformed — none of those
sentences mention the environment, and all of them were the environment.

So the first question when something fails is not "what is wrong with the
target" but **"has this host ever done this successfully?"**

## How to use this

Entries are keyed on **verbatim output** — the string you actually saw, which
is what you can grep. That is deliberate and it is the same rule the catalog
follows: a quoted diagnostic survives a refactor, a line number does not.
Search this file for the error text, not for a description of it.

Each entry says whether the workflow **already catches it**. If it does, you
should never have reached the symptom, and reaching it anyway is itself worth
reporting — a check that has stopped working is worse than no check.

## Preflight catches most of this before an agent exists

Run `--dry-run` first. It resolves the environment and stops before the first
agent turn, and it is the cheapest possible answer to "can this host host this
run".

| check | what it proves |
|---|---|
| `check_checkpoint` | the checkpoint parses and declares an architecture |
| `check_tmpdir` | `TMPDIR`/`CLAUDE_CODE_TMPDIR` are set and not under `/tmp` |
| `claim_devices` | a local run: enough idle devices of the declared arch, by free memory and a lone-process probe. Under a scheduler the scheduler hands out devices |
| `check_multirank_substrate` | NCCL collective + MPI spawn answer under a deadline, for a multi-rank run |
| `probe_agents` | in-process turns: every agent the roles run on takes a turn here and its **Bash tool runs a command**, judged by the file it writes |
| `substrate_gate` | **this host generates a token** on the stock path under this run's exact runtime block |

`substrate_gate` is the one that earns its minutes. It *generates* rather than
constructs, because the NVRTC failure below appeared on the first generated
token — after the engine had loaded 22 GB of weights, allocated the KV cache
and finished warm-up. A gate that stopped at "the engine constructed" passes
it straight through.

A resumed run (a reload, a relaunch) does not run it again when its last pass,
`<workspace>/gates/substrate/pass.json`, is under 12 h old and was recorded for
the same resolved environment, checkpoint, runtime block, options file and
checkout path; the journal says `substrate gate not run` and why. A repair the
record cannot see (an image re-imported under the same name, a package
installed into the same interpreter) needs that file deleted before the restart.
The agent probe's pass, `<workspace>/gates/agent-probe/pass.json`, is reused the
same way, for the same agents, CLI, host and login; a `--dry-run` runs the probe.

---

## Launch substrate

### `MPI_ERR_SPAWN: could not spawn n processes`

**Not a model problem.** The container runtime forbids `MPI_Comm_spawn`, and
the default LLM API path uses it. Under pyxis/enroot this is the normal state
of affairs, not a misconfiguration.

The decision is one line, in `llmapi/mpi_session.py`:

```python
def need_spawn_mpi_workers(model_world_size: int) -> bool:
    return get_mpi_world_size() == 1 and model_world_size > 1
```

**So the fix is to make the ranks exist before the engine starts**, which is
what `trtllm-llmapi-launch` is for: one task per rank, rank 0 runs the leader
and the real task, the rest run workers and wait. The workflow emits this
automatically at world size > 1 — an allocation sized without `--ntasks`
sized to match is the whole bug.

**`TLLM_WORKER_USE_SINGLE_PROCESS=1` is not the general fix.** It works, and
this run used it, but the function behind it is named
`enable_worker_single_process_for_tp1` and the source comments call it a WAR
for TP1. It does not scale past one rank. If you find it in a spec at world
size > 1, that spec is wrong.

At world size 1 the right answer is the other branch: no wrapper, and clear
the MPI environment before exec. mpi4py auto-initializes MPI whenever PMI
variables are present, and inside a PMIx-managed step they always are — so a
single-task job initializes MPI for no reason and then dies spawning a worker.

*Caught by:* `substrate_gate`. Note that `check_multirank_substrate` does
**not** catch it: that probe uses `torchrun`, a different launcher, so it
proves NCCL works between local processes and says nothing about whether the
gate's own `srun` line can start a multi-rank engine. It goes green first,
which is misleading.

### The engine hangs at startup and never returns

A wrong launch substrate **hangs rather than raises** — which is why the
multi-rank probe has a deadline at all. The usual cause is an MPI connect-back
landing on an interface that does not carry it. Without the deadline this
surfaces as "the engine never booted", hours in.

*Caught by:* `check_multirank_substrate`, which fails on the timeout rather
than waiting.

### `There are not enough slots available in the system`

In full: *"There are not enough slots available in the system to satisfy the 4
slots that were requested by the application: trtllm-llmapi-launch"*, before
any Python runs.

Open MPI takes its slot count from the resource manager, and a `session_spool`
job is submitted as **one task** that owns the node (`NumTasks=1` against
`NumCPUs=144` and `gres/gpu:4`). So `-n 4` asks for four slots against one.
The allocation really does hold the whole node, so the slot count is a
bookkeeping guard rather than a real constraint.

Do **not** fix it by raising the session's `ntasks_per_node`: that gives MPI
its slots and simultaneously starts four copies of the spool watcher, which
then race for every piece of work.

*Caught by:* `_mpirun_launch`, which adds `--oversubscribe`.

### Four `mgmn_leader_node` processes, and a log that stops growing

No error, no traceback, no exit — the shape above, with a specific cause. A
correct 4-rank launch has one leader and three workers; four leaders means
every rank thinks it is rank 0.

`trtllm-llmapi-launch` derives rank as
`mpi_rank=${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-...}}` and reads
`SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE`. Slurm wins over Open MPI in
both. That is right under `srun` and exactly wrong under `mpirun` *inside* a
Slurm allocation, where `SLURM_PROCID` is 0 for every rank and `SLURM_NTASKS`
is 1. Measured, four ranks: `SLURM_PROCID` read `0, 0, 0, 0` where
`OMPI_COMM_WORLD_RANK` read `0, 1, 2, 3`.

MPI itself is not at fault — a bare `mpi4py` probe under the identical
`mpirun` line reports `rank 0..3 of 4`.

*Caught by:* `_mpirun_launch`, which prefixes `env -u SLURM_PROCID -u
SLURM_NTASKS ...` so the launcher falls through to the Open MPI variables.

### `UCX ERROR uct_iface_open(tcp/rdma_vf_rail0) failed`

Noise, on every rank, and a documented red herring. UCX fails to open one IPv6
RDMA virtual-function interface, falls back, and the run completes with exit
code 0 and correct output. `UCX_TLS=sm,self` silences it and fixes nothing.

*Not caught, and deliberately not.* If the run is also hanging, the cause is
the entry above, not this.

### Two jobs die together, both at `MPI_Init`

Concurrent `MPI_Init` in the same allocation killed two compute sessions in
one run. The spool scheduler removes the allocation's MPI context from every
job, which addresses the likely mechanism; if it recurs anyway, serialize the
jobs that initialize MPI (`--gpus all`) or give them distinct allocations.

*Not caught.* Observed; the mechanism was not isolated further.

### Setting `TLLM_DISABLE_MPI=1` makes it worse

It routes the executor to Ray, which was not installed, so the failure moves
rather than going away. Do not reach for this as a workaround for the spawn
error above — `TLLM_WORKER_USE_SINGLE_PROCESS=1` is the one that works.

---

## The engine boots but will not run

### `Impossible to fit in any sequence in kvCache`

The engine inherited the checkpoint's `max_position_embeddings` — 262144 is
common — and sized the KV cache for it. Nothing is wrong with the checkpoint
or the target.

Declare the runtime explicitly in `task.yaml`'s `runtime.extra_flags`:

```yaml
runtime:
  extra_flags:
    - --max_seq_len=8192
    - --max_batch_size=8
    - --kv_cache_fraction=0.6
```

*Caught by:* `substrate_gate`, which runs under the same `runtime` block the
gates will use — so a bad sizing fails in preflight rather than mid-assembly.

### `NVRTC ... could not open source file "cuda.h"`

**Two independent things produce this, and one of them is the boring one.**

1. **The venv's `bin` is missing from `PATH`.** NVRTC needs the toolkit
   headers and finds them relative to the interpreter's environment. This was
   the root cause in this run, and it was found late because the symptom
   points at CUDA rather than at `PATH`.
2. **The generation FMHA kernel takes the NVRTC path.** In
   `kernels/trtllmGenKernels/fmha/fmhaKernels.h`, the kernel is selected for
   NVRTC unless the KV dtype is `E2m1` — grep for
   `options.mDtypeKv != tg::Dtype::E2m1`. On a build whose NVRTC has no header
   search path, that path dies compiling. Setting `--kv_cache_dtype fp8` takes
   the non-NVRTC branch and the error goes away.

**A wrong diagnosis worth recording:** rebuilding with
`--nvrtc_dynamic_linking` (NVRTC_DYNAMIC_LINKING OFF→ON, 98% ccache hits) did
**not** fix it. And the `fp8` workaround, while it genuinely works, works for
the second reason above, not the first — so "fp8 fixed it" is a true sentence
with a false explanation attached. If you hit this, check `PATH` before you
rebuild anything.

*Caught by:* `substrate_gate` — but only because it generates a token. The
fault is on the first generated token, not at construction.

### The recurrent state runs at the wrong precision, silently

A checkpoint declaring `mamba_ssm_dtype: float32` still gets the engine flag's
default of bfloat16. Nothing errors; the model is just half as precise as it
was calibrated for. Pass `--mamba_ssm_cache_dtype=float32` in
`runtime.extra_flags`.

*Not caught.* This is the class the **surface survey** exists to surface: the
checkpoint declares it, so it appears in the survey's demands at plan time.

---

## Cluster access

### `sinfo` / `sbatch` / `squeue` cannot contact the controller

On some clusters the Slurm CLI is simply unreachable from the session, and an
MCP broker is the only path. **Do not conclude the cluster is down.** Use
`environment.kind: session_spool`: the driver session starts each compute
session through the broker (`trtllm-modeling-v2 session submission` prints the
call), and everything after that is files.

### A broker job "fails" after exactly 300 seconds but is still running

The MCP layer aborts a tool call after 300 s of silence. Every real job here is
longer than that, so the blocking `slurm_run` reliably reports failure for a
healthy job and leaves it running with its output unreachable.

Use `slurm_submit`, poll `slurm_job_status`, then read the log file. Never the
blocking call.

### Work started in a job step dies when the step ends

`nohup` does not save it. Work started inside a `slurm_exec_in_job` step does
not survive the step's teardown — measured.

This, plus the 300 s cap, plus a 26 GB container image costing 4–8 minutes to
import **per job**, is why `session_spool` exists:

- one allocation, acquired once, running the spool scheduler as its own main
  process (see *Compute sessions and the spool* below);
- **work is a file** — drop `pending/<name>.sh`, wait for `done/<name>.rc`,
  read `logs/<name>.log`;
- so anything that can write a file can run GPU work: the orchestrator, an
  agent, or a human.

### `invalid_mount ... world-writable governing directory`

```
{"error":"invalid_mount","status":400,
 "detail":"mount source has a world-writable governing directory
           /.../artifacts (post-validation component-swap vector)"}
```

Nothing is wrong with the path you asked for; something **above** it is
world-writable. CI artifact trees usually are (`drwxrwsrwx`), so a checkpoint
published there cannot be mounted where it lives and has to be staged into a
directory you own.

Staging has its own trap, and it reads as a quota problem: the orchestrator's
sandbox grants write access only to its **declared working directories**, so a
copy into `<user_root>/models` fails with `Read-only file system` even though
the permission bits are fine. Stage inside the checkout and exclude it via
`.git/info/exclude`.

### A job submits fine by hand but the workflow's submission is rejected

Check the GPU request form. NVL72 policy rejects job-scoped `--gpus`;
`gpus_per_node` is the only form that submits there. This is a cluster policy,
not a workflow bug — `environment.gpus_per_node` exists for it.

---

## Compute sessions and the spool

A session is an allocation whose main process is the spool scheduler
(`infra/substrate/scheduler.py`). Every job's verdict is in `done/<name>.json`, and
its `reason` names what happened. `spoolctl.py status` shows what runs, what
waits and why.

### The safe-kill rule, and a whole allocation that died when one job did

Six session deaths in one run had one cause: a process that inherited the
allocation's Slurm/PMI context died abnormally (a crash, a timeout, an
operator's `kill`), and Slurm's MPI error handler (`_errhandler status=-61`)
killed the **whole allocation** — scheduler, orchestrator and every queued job.

**The safe-kill rule:** never signal a process that carries the allocation's
MPI context (`SLURM_*`, `PMI_FD`, a `slurm.pmix.*` `PMIX_NAMESPACE`). The
scheduler removes that context from every job it starts, so it can stop its
own jobs safely; it still checks each process's environment before signalling
it, and a process that carries the context is left alone: its job is marked
failed (the waiter unblocks) and its devices are quarantined until it exits by
itself. To stop a job, use `spoolctl.py cancel <name>`. Never `kill` a job's
processes by hand, never `pkill`/`killall` by pattern, and never kill the
scheduler: it is the allocation.

*Caught by:* `scrub_environment` at launch and `is_mpi_attached` before every
signal. `trtllm-modeling-v2 session watchdog` reports quarantined devices.

### `no watcher heartbeat in <spool>: the session is not running` (exit 70)

Either the session is down, or it is watching a different directory. The
second one reads exactly like the first: a hand-edited entry script copied
from another run came up on that run's spool, and every submitter reported a
dead session. Compare `spool=` in the spool's `watcher.status` with the path
you submit to, and render entry scripts instead of editing them
(`trtllm-modeling-v2 session render`).

*Caught by:* `queue_job`, which refuses a spool with no live heartbeat.

### A job waits while devices sit idle

Read the reason in `spoolctl.py status`:

| reason | meaning |
|---|---|
| `waiting for the whole node to drain` | a whole-node job is next; slot jobs are not admitted until it has run |
| `queued behind <job>` | a GPU job ranked above it is waiting for devices it will need |
| `host memory N GiB available, M needed` | it declared `host_mem_gb` (whole-node jobs inherit `engine_host_mem_gb`); the previous engine's host memory is still being released |
| `device memory in use: <dev> (N MiB) > M MiB` | a device it would get still holds memory no running job of this session accounts for: an ended job (a rank, a launcher child) has not given it back. After `device_release_wait` the device is quarantined (`quarantined, memory not released: ...`) and the job is placed without it |
| `native rebuild of <tree> in progress` | a rebuild was announced on this session, of the job's tree or (`<job tree> shares its native build`) of the checkout whose native build the job's snapshot or worktree links to |
| `... waits for the successor session` | the allocation ends before the job's wall limit would |

*Caught by:* `_admit`, which writes the reason into the status file.

### A KV-cache pool or slot count smaller than the same engine's elsewhere

An engine sizes its KV-cache and recurrent pools from the device memory it
finds free at boot. Started beside memory a cancelled job (or one of its ranks
or launcher children) had not given back yet, an mtp2 tp1 boot reserved 642
cache slots where the identical flags reserve 1237, and its gate certified the
smaller pool. Before a GPU job starts, the scheduler reads its devices with
`nvidia-smi` and holds it while one has more than `device_max_used_mib`
(1024 MiB by default) in use; a device still over it after
`device_release_wait` (5 minutes) is quarantined until a reading finds it
released, and the watcher's `watcher.status` says when it went and came back.
If you see the symptom anyway, check that `nvidia-smi` runs in the session's
container: without it nothing waits.

*Caught by:* `_place`.

### `reason=wall_limit` (rc 124)

The job ran past `--wall`, counted from its start: time queued behind other
work is not charged. A gate's timeout is its wall limit.

### `reason=stalled` (rc 210): `no progress for Ns after a failure line`

The log showed a failure (a traceback, `MPI_ABORT`, a segfault, a CUDA or NCCL
error) and then nothing for `failure_stall` (10 minutes by default for GPU
jobs). That is the shape of a rank-0 init failure with the other ranks waiting
in a collective forever. Read the failure line in the receipt's `detail`; the
rest of the log is the ranks that waited.

*Caught by:* `_track_progress`.

### `reason=mixed_checkout` (rc 217)

The job's processes imported the engine package from more than one checkout,
or not from its `--tree`. An MPI worker imports from its working directory
ahead of `PYTHONPATH`, so pointing only `PYTHONPATH` at a worktree runs the
worktree on rank 0 and the main checkout everywhere else — and the result
describes neither. Submit with `--tree <worktree>`, which sets both. Every
process's import is in `logs/<name>.resolve.jsonl` and in the
`[mv2flow-resolve]` lines of the log.

*Caught by:* `_judge_provenance`.

### `reason=memory_limit` / `memory_pressure` (rc 211)

The job exceeded its own `--mem-gb`, or host memory fell below the session's
`pressure_floor_gb` and this job held the most. Twelve concurrent test jobs
once pushed a session's memory cgroup over its limit and the whole allocation
was killed; this stops one job instead.

### `reason=host_memory` (rc 214)

The memory the job declared never came back within `host_mem_wait`. The
receipt names the largest jobs on the session; if none holds it, a process
outside the scheduler does.

### `reason=orphaned` (rc 212)

The scheduler that ran the job stopped heart-beating, but the job's processes
still run on this host, unattended (`detail` names their pids), or no watcher
recorded starting it. A scheduler on the same spool closed it out rather than
run it again beside itself. Nothing in it is a verdict; once those processes
are gone, run it again.

### `reason=node_lost` (rc 218)

The allocation the job ran on is gone (Slurm requeued it onto another node,
or it ended or failed) and the job went with it, and it was not run again:
it asked not to be (`requeue=0`), it had already run again as often as it may,
or its cancel was requested; `detail` says which. The receipt's `attempts`
lists every allocation it ran on and when each run was lost. Nothing in it is
a verdict; run it again.

*Caught by:* `_lost`.

### `spoolctl: <name>: ...; queued again for run 2 of at most 2`

The allocation running the job was lost, and the next scheduler on the spool
(the requeued start, or a successor) queued the job again, keeping its place
in the queue. `spoolctl.py submit` and `wait` go on waiting and return the
new run's receipt, whose `attempts` lists both runs; the cut-off run's log is
under `history/`, and `spoolctl.py status` lists such jobs under `queued
again after a lost allocation:`. Do not resubmit it: that runs it twice. A
job runs again once unless its header says otherwise (`spoolctl.py submit
--requeue N`, header `requeue=N`; `requeue=0` closes it out `node_lost`
instead); the scheduler's `--requeue` sets the session's default.

*Caught by:* `_recover_orphans`.

### `the session watching <spool> stopped heart-beating ...; <name> is still in running/` (exit 70)

`spoolctl.py submit` or `wait` stopped waiting because the session stopped
heart-beating, as it does for the minutes a requeued allocation takes to come
back. The job is not lost with the wait: the next scheduler on the spool runs
it again or closes it out `node_lost`, and `spoolctl.py wait <name>` returns
that. `--session-wait 15m` on `submit` or `wait` waits that long for a
session instead.

*Caught by:* `left_in_spool`.

### `reason=cancelled` (rc 216), `reason=refused` (rc 213), `reason=launch_failed` (rc 215)

`cancelled`: someone ran `spoolctl.py cancel <name>`; the scheduler stopped the
job under the safe-kill rule, or never started it. `refused`: the job could
never run here, and `detail` says why: a header key it does not know (a
misspelt limit is refused rather than ignored), more GPUs than the session
hands out, or a `--tree` the compute node cannot see. `launch_failed`: the
scheduler could not start `bash` for it at all, which is a node problem.

### `job name ... must start with this requester's prefix`

Each external turn gets a prefix (`r014` for request `014-smith`), and the
submit script refuses names outside it: two turns once picked the same name
and one deleted the other's receipt. `export MV2FLOW_JOB_PREFIX=<prefix>`
and name jobs `<prefix>-<what>`.

*Caught by:* `check_name`.

### A process faulted (SIGSEGV) about a minute after someone rebuilt the native libraries

`build_wheel.py` rewrites `tensorrt_llm/libs/*.so` in place, under every
process that has them mapped, on every node. Rebuild only through the
workflow's `incremental_build.sh`, which gives every library a fresh inode
first and holds new GPU jobs against that checkout on every session until the
build ends.

*Caught by:* `refresh_inodes`.

### `... did not run to a verdict: ...`, and a run closed `session-lost:<stage>`

The gate's compute session stopped under it: the allocation was requeued, or
ended before its successor started. The gate measured nothing, so nothing is
recorded and no assembler turn is spent. The orchestrator withdraws the job,
waits up to `environment.session_wait` (default 1 h) for a session to
heart-beat on the spool again, and re-runs the gate, at most three times in a
row. Past either bound the run closes `session-lost:<stage>`. Bring a session
back (`trtllm-modeling-v2 session status`, `session allocate`) and re-run the
same command; it resumes at that gate. A session that keeps coming back and
dying is a substrate to repair, not one to outwait.

*Caught by:* `_gate_until_verdict`.

### The allocation reached its time limit in the middle of a gate

Declare the session's `time_limit`. `handover_lead` before the end, the
scheduler writes `handover/request.json`, stops admitting jobs that could not
finish, and leaves them in the spool for the successor; `trtllm-modeling-v2
driver wait` returns `SESSION_REQUEST`, and `trtllm-modeling-v2 session allocate`
prints the call that starts the successor on the same spool, after the
allocation its live watcher holds, and the `session ack ... --after <job>` that
records it. A successor submitted and acknowledged before the request is
written already answers it: the ack, `handover/acks/<allocation>.json`, is
keyed by the allocation the job follows.

*Caught by:* `_plan_handover`.

### `SESSION_REQUEST` for an allocation that ended a day ago

A request left behind by an earlier hand-over, answered by an ack that a later
`session ack` then replaced. Both halves are gone: an ack is a file per
allocation (`handover/acks/<allocation>.json`), the successor moves its
predecessor's request and acks to `handover/archive/` when it starts, and
`driver wait` archives a request whose watcher no longer heart-beats and says
so after its event:

```
TIMEOUT  nothing new
  note: session g3: archived request g3-1000027-1792258324 (allocation 1000027): allocation 1000027 has no live watcher: it ended
```

Nothing is asked of you for it. `its successor <watcher> ... is up` and
`stopped heart-beating; allocation <job> runs on under a new watcher (requeued
or restarted)` are the same: the request is moot. Do not archive requests by
hand; `session requests` names the ones it ignores.

*Caught by:* `stale_request` in `driver wait`; `_take_over_handover` when a
successor starts.

### `session allocate` prints `dependency="afterany:<job>"` for an allocation that has ended

A dependency on an ended job is met at once: the "successor" starts beside the
allocation that holds the spool, and the session runs two allocations.
`session allocate` takes the job from the session's live watcher (the one that
ends last), never from the request on disk, and says which:

- `It starts when allocation <job> ends (...)`: submit it;
- `No dependency: no watcher on <spool> has heart-beat within 180s ...`: the
  session is down, and the call starts it as soon as Slurm schedules it;
- `Allocation <job> already has a successor acknowledged: job <id> ...`: one
  is queued; submit another only if that one is gone.

*Caught by:* `plan_successor`.

### `orchestrator: unknown (... writes no heartbeat (turns run in process) ...)`

An orchestrator on the in-process transport keeps no `dispatch/` heartbeat.
`driver wait` reads it from the run's ledger (`inflight.json`) and the
workspace lock: a lock held is a running orchestrator from any host; a run that
closed says so in the ledger; a lock free is an exit only on the ledger's host,
because another host may not see the lock. Run the operator loop (`driver
wait --sessions-only`) on the host the detail names, and it reports a crash or
a kill as `ORCHESTRATOR_EXITED` too.

*Caught by:* `_ledger_state`.

### The driver waits and nothing happens

`trtllm-modeling-v2 driver status` shows the orchestrator's state. `exited`
means a run that closed, crashed or lost its allocation: its log says which,
and a relaunch adopts every unanswered turn, and every answered turn of a task
that had no outcome yet; `driver status` marks those `ADOPTED`, so an answer
already given is not asked for again. Do not copy an old decision to a new
request's path by hand. `CLAIMED ... OVERDUE` means a turn ran past its role's
timeout and its subagent has likely died: `release` it and dispatch it again.
`reload: draining (...)` means a reload is waiting for what it names, and
starts nothing until then; see the next entry.

*Caught by:* `orchestrator_state`.

### A flow fix has to reach a running run, and a relaunch would cut work short

Do not kill the orchestrator to load it. A stop by hand cuts short whatever is
in flight: a rebase (its commit lost), gate jobs (re-run, and their devices'
memory held until the scheduler releases it), a planner answer about to be
read. Run `trtllm-modeling-v2 driver --workspace <ws> reload`: the orchestrator
starts nothing new, waits for the integrations, checks and planner turn in
flight, checkpoints and re-executes itself on the code now on disk (README,
[Reloading and relaunching](README.md#reloading-and-relaunching)). Worker turns
go on, and the new process adopts them. `driver wait` returns `RELOADED` once
its loop runs again.

- `RELOAD_OVERDUE`: the drain passed its `--timeout` and names what it still
  waits for. A check or an integration usually ends on its own; wait, or
  `driver reload --force` to re-execute at the deadline without it (a cut
  check runs again), or `driver reload --cancel`.
- `RELOAD_FAILED`: the re-exec itself failed (the reason is in `reload.json`
  and the journal); the run goes on in the old process, on the old code. Fix
  the cause and ask again.
- `ORCHESTRATOR_EXITED` right after a reload: the new code did not start (an
  import error, say). Its log says why; fix it and relaunch the same command.
- The journal's `restart (hard-stop)` line after any start names what the stop
  before it cut short, and `restarts.jsonl` keeps one line per start.

*Caught by:* `_reload_check`, which drains; `_record_restart` records what a stop cost.

### `Agent stalled: no progress for 600s (stream watchdog did not recover)`

Too many concurrent turns on one account: seven of twelve died this way within
twelve minutes. `driver wait` hands out at most 16 worker turns at once
(`DEFAULT_MAX_INFLIGHT`, `--max-inflight`), whatever `--max-parallel-tasks`
(default 16) lets the orchestrator ask for; the rest wait as requests. The
planner's turn and the critic's read come beside that cap, one of each at a
time. If the stalls return, lower both with `--max-inflight` and
`--max-parallel-tasks`.

---

## A task derived from the checkpoint (`--checkpoint`)

### `✘ intake: N block(s) in <workspace>/task.intake.yaml`

Phase 0 ran the request's checks before touching the host, and every block is
listed (the journal has them under `intake checks failed`). Each is a fact
about the request, the checkpoint or the compiled spec, not the host: an open
question the derivation could not answer (`device.arch`, a witness's
`contrast_env`), members a checkpoint's heads or experts do not split across,
an anchor row the reference file lacks. Answer them in the request, or in the
goal or `site.yaml` with `--clean`, and re-run the same command; a run that
never got past preflight takes an edited request as it is.

`accuracy anchor [<unit>]: ... has no entry matching {'quant_algo': 'NVFP4'}`
is the anchor check keeping a quantized checkpoint off the unquantized row:
add the row the checkpoint's own format names, citing where the number came
from, or set `accuracy.discriminators` in the request.

### `the request changed after its run started from it`

The run is past preflight and `task.intake.yaml` no longer has the digest in
`task.intake.yaml.started`. An edit after the run started is a decision about
work already done: confirm it (`--confirm` on the command, or
`trtllm-modeling-v2 intake confirm <request> --by NAME`) and re-run, or start
over with `--clean`. `changed after it was confirmed` is the same for a request
someone confirmed.

### `the workspace's request was derived for the goal ...` / `... from <checkpoint>`

The workspace already holds a request, and the command asks for another. The
request on disk is the run's and may carry edits, so it is never re-derived
silently: edit it, use another `--workspace`, or `--clean` to derive it again.

### `argument --goal: expected one argument`

A goal that starts with `-` (`-mtp tp1`) reads as an option. Write
`--goal=-mtp tp1`, or `no-mtp tp1`.

## The planned run

### `MISSING <unit>:<gate>@<member>` for a check that passed before

A verdict holds for the state of the target its check measured: a
fingerprint of the commit the checkout was at, whose snapshot the check ran
on, less what no gate runs or reads. An uncommitted change in the checkout is
in no snapshot and moves no verdict. A commit
under `tests/`, `docs/`, `.github/` or `examples/`, or to a `*.md` file,
leaves every verdict standing, unless it changes a file a gate reads there:
the quickstart, the default boot image, or a file the spec's `accuracy`,
`runtime`, `variants` or `environment` block names. Of an anchor table
(`reference_yaml`) only the entries of the model keys the run reads count,
unless the spec names the table elsewhere too. Any other change, anywhere
in the tree, makes every verdict on the target stale, and so does a native
rebuild: the libraries in `tensorrt_llm/` and `tensorrt_llm/libs/` are outside
git, so their sizes and modification times are part of the fingerprint. The
checks a finished task claimed run again once no task is running.

`SHARED <check>: holds by the pass of <other>` in the same standing is not
missing: the two run one command on one engine against one criterion, so
one pass on this tree serves both. A failure is never shared.

### `[VOID] <gate>@<member>`, or `VOID <check>` in the standing

The check ran on the snapshot of one commit (`<workspace>/snapshots/<commit>`),
but a command printed that it imported the package from somewhere else: the
verdict names it (`imported` in `contract_status.jsonl`, one entry per command
that said). A path is a checkout no commit pins, usually the integration
checkout itself; two commits are two snapshots. The check was measured once
more on the same snapshot before it was recorded as not passing. Find what put
the other tree ahead of the snapshot on `sys.path`: a `PYTHONPATH` in the
spec's `environment.env`, a leg B launcher (`contrast_command`) that changes
directory or sets its own `PYTHONPATH`, or a package installed into the
interpreter rather than linked to a checkout. A void verdict that says no
snapshot could be made carries git's message; `git worktree list` in the
checkout shows the snapshots that exist.

### `the checkout holds ... uncommitted; this check measured <commit> without them`

A warning, not a failure: someone edited the integration checkout instead of
a task worktree. The check measured the commit; commit the change (through
`bin/commit.sh`) to have it measured, or remove it.

### `bundle: <check> runs singly: ...`

The check was admitted into the gate bundle of its unit and member, and is
measured alone instead; the line says why, and it is never a verdict:

- `its engine differs from the bundle's: <setting>: quickstart X, trtllm-eval Y`
  (an accuracy check): the accuracy gate's `trtllm-eval` would build another
  engine than the boot gate's quickstart. Usually the unit states a limit in
  `runtime.extra_flags` and another (or none) in `accuracy.max_batch_size`,
  `accuracy.max_seq_len` or the options file. State each once, the same in
  both, and the next round bundles it. An options key or a quickstart flag
  the comparison does not read (`` `attention_dp_config` in <file>``,
  `quickstart flag --moe_tp_size`) keeps the evaluation alone by design.
- `its section failed in the bundle: <error>`, `its section ran past its own
  time in the bundle`, `the bundle ended inside its section (...)`: the
  workload failed on the shared engine. Its single run gives the verdict; the
  checks after it in the bundle run singly too (`an earlier section (...)
  ended the bundle`), and those before it keep the verdicts they got.
- `the bundle did not run to an end: <reason>`: the bundle's session was
  lost; every member reruns alone, each waiting for a session as any gate does.
- `the engine did not run the declared configuration -- ...` (an accuracy
  check): the engine the bundle built is not the one the unit's options file
  declares on that field, so the evaluation builds its own.
- `no other check of its bundle builds its engine`: a check without the
  speculator beside checks with it, say; nothing is wrong.

*Caught by:* `split`.

### `failed on attempt 3 of 3 and is not retried again`

The planner wrote a failed task back as `pending` once more than it may. A
retry runs the task again under its own id as its next attempt (`attempts`
in `graph.yaml`), and the third failed attempt is the last: the revision is
refused, and the planner is told to find the cause first (an `investigate`
task), reshape the work, or drop the task. A failure of the run's
infrastructure (the turn raised, `INTEGRATION_FAILED`, a tune baseline that
could not be measured) costs no attempt: the task starts again once by
itself, and the planner hears why.

*Caught by:* `apply_revision`.

### `INTEGRATION_FAILED`: `conflicts with the integration head ... in: <files>`

The task finished, but its branch did not rebase onto the checkout's head,
and its worker, sent back twice to rebase onto the head it met and resolve
(`back to its <role> to rebase onto <head> and resolve` in the journal),
could not. The task starts once more by itself, as an infrastructure
failure; after that, merging its branch is an `integrate` task the planner
adds. Two tasks that edit one file conflict this way: name files in
`touches`, so the scheduler keeps such tasks apart.

*Caught by:* `rebase_block`.

### `geometry probe ... 0 of N leg(s) measured; not measured: ... does not know this checkpoint's architecture`

The stock path cannot boot a new architecture, so the probe measured nothing
before the kernels, and the first certifications run on estimates again.
Declare `environment.substrate_checkpoint` (a checkpoint the stock path
supports): each leg then boots it under the same runtime block and keeps the
capture list, the limits, the page size and the widths, which the
configuration decides. The pools and chunk splits stay unmeasured until the
target's first gates pass. A leg `failed` for another reason (`exited 1`, a
timeout) is the stock engine at that width: the substrate gate's notes and
`gates/geometry/geometry.<unit>.<member>.json` reproduce it.

*Caught by:* the geometry probe's journal line; it never stops the run.

### A certification closed as covered that should have run

`certification closed as covered` in the journal: every cell the dispatch
stated is a cell of the entry's `CELLS` at the checkout's `HEAD`, on the
run's architecture. The decision file's `covered` lists each required cell
and the cell that covered it. If the work needed is not a cell the entry
lists -- a shape no spec key states, a new call site -- the row's `cells` (or
the goal's ```` ```cells ```` block) does not say what the certification is
for: state the missing cell, or leave `cells` off and the smith is
dispatched. `certification dispatched: <why>` says why a stated one was not
closed.

*Caught by:* `run/covered.py`; the entry's own test holds the covering cells.

### `[FAIL] ops:promoted: <path> is a local variant of <category>/<name>`

Delivery found a local variant still in the target's `ops/`, or a target
file still importing one (`<file> imports ops/<category>/<name>__<site>`).
Kernel tasks on a shared entry land variants so they can run side by side;
the `promote` task merges them into the catalog once the target is otherwise
complete. Reaching this means no promote ran after the last variant landed:
the last one failed (the orchestrator adds no other after a failure; the
journal's `promote` task says why, often two variants that changed
`__call__` differently, or a cell that passes alone and fails merged), or a
variant landed after it. Retry or reshape the promote, or, when the target
really should keep its own copy, name the entry in `task.yaml`
`target.keep_local`. A task that imports a variant after the promote removed
it fails at its first gate with an `ImportError` on the variant's module:
point that import at `catalog/<category>/<name>`.

*Caught by:* `check_promoted`.

### `every required check holds, but no cross-model regression review covers <tasks>`

Every check passes and nothing required is open, yet the run is not
delivered: those tasks landed changes to shared code (outside the target's
own paths) that no regression review's batch covers. Usually the review that
would have covered them failed (a `REJECT` covers nothing), or they landed
after the last one started: a retry that landed again, a kernel row the
orchestrator added, a failed receipt whose guard commits went in. The planner
hears this line and adds, or retries, a required `review` with
`regression: true`; as it starts it covers every such change at once.

*Caught by:* the orchestrator, before it delivers.

### A regression review whose dependencies are done does not start

It waits on purpose, and the journal's `[regression]` line says what it took
when it did start. One regression review runs at a time, and the others wait
for it; while a task that changes shared code runs, a ready one waits up to
30 minutes after its batch's first landing, so that change joins the same
review. `task <id> (review) closed with no turn: no change to shared code was
left for it to review` is not a failure either: the reviews before it covered
every change it follows.

### `the engine's greedy tokens are not the HF reference's (<file>): <prompt> step <n>: engine token <a>, reference <b> (...)`

The boot sent the logit reference's text prompt as token ids and the engine's
greedy token at step `<n>` is not the reference's, and not a near-tie: the
reference's own top-1 leads the engine's token by more than the logit gate's
largest |delta log-prob| limit, or the engine's token is not in the
reference's top-k at all. Read it as the first token a defect changed, not as
noise: a wrong layer, a misread weight, a wrong position, a speculative verify
that accepts what the target would not have generated. The steps before `<n>`
agreed, so the cause is in what that position computes. Run the logit gate at
the same member (`trtllm-modeling-v2 gate logits`) for the per-position
numbers, and the same boot under `--mode off` to see whether the stock path
agrees with the reference there.

`generated <k> of <n> tokens` is an engine that stopped before its budget: the
boot asks for every token past any end token, so a short answer is a dropped
request or a length limit, not an end of sequence.

*Caught by:* `gates.judge_greedy`.

### `the boot's greedy reference: ...`, or `no text prompt of the greedy reference ... fits this unit's engine`

The run names a logit reference (`accuracy.logits.reference`, or
`logits.reference.json.gz` beside the spec), so every boot compares the
reference's greedy tokens, and this boot could not: the file is missing or not
a reference, or every text prompt in it is longer than the unit's
`max_num_tokens` without chunked prefill, or than `max_seq_len`. Make the
reference (README, *Logits*) or one with shorter prompts, or set
`accuracy.boot.greedy.enabled: false`. With the comparison off, a unit whose
target runs a speculator keeps its benchmark at every member.

*Caught by:* `gates.greedy_workload`.

### A unit's benchmark is owed at fewer members than it has

With the logit gate owed, `accuracy[<benchmark>]` runs only at the narrowest
member, the widest, and every member the logit gate cannot stand for
(attention DP; a speculator whose boot compares no tokens); the other members
owe `logits`, which holds every position to the reference. `contract.yaml`'s
`notes` say where each unit's benchmark runs and why. Not a missing check:
`accuracy.benchmark_members: every` asks for it at every member.

---

## Performance measurement (the tune stage)

### `TUNE_FAILED (tune-failed:smoke)`: `the perf smoke failed; the baseline is not started`

Before its baseline, the tune task serves each world size its perf configs
use once (concurrency 1, 4 prompts of 32 output tokens, a 45-minute wall,
`perf.campaign.smoke_wall`) on the perf sessions. The line names the world
size and its perf config, then either the job's return code and log
(`rc=<rc>, see <log>`) or ``served `modeling_v2: off`: the perf config does
not select the target`` — a serving config without `modeling_v2: require`.
A launch that hangs, ranks left on the devices or a device another job holds
fail here in minutes instead of after a whole baseline. Fix the launch or the
config, then retry the task; `perf.campaign.smoke: false` skips the smoke.

*Caught by:* `smoke`.

### `[tune] <label> on disk ...; set aside as <label>.stale-<time> and measured again`

Not a lost result. A measurement on disk is reused only when its `meta.json`
says it measured what the campaign measures now: the same workload, the same
merged LLM-API config and, on the campaign's own tree, the same commit. A
spec or flow change between two attempts of the task, or a best tree that
moved since, leaves the old one beside the new under `.stale-<time>`
(`stale_measurement` in `perf_ledger.jsonl`).

### nsys: `Profiling started at iteration N`, then the server sits at 0% GPU and no report is written

Two causes, stacked, on a single-process `trtllm-serve`:

- `torch.cuda.cudart().cudaProfilerStop()` (what `tensorrt_llm/_torch/pyexecutor/profiling.py`
  calls) holds the GIL while CUPTI waits for GPU work that other Python threads must complete:
  gdb shows every other thread in `PyGILState_Ensure`.
- Calling the driver's `cuProfilerStop` through ctypes instead still did not return with
  `-c cudaProfilerApi`, with node- and graph-level graph tracing alike.

Use `bench.py --nsys <prefix>` in its default session mode: `nsys launch` + `nsys start/stop` timed from
the measured point (`--nsys-delay`, `--nsys-duration`); nothing in the server calls the profiler API.

### A measurement shares its GPU with a server nobody is running

A cancelled job's `nsys profile` (or any server started in its own session) is re-parented to init and
outlives the job's leader. The scheduler finds such processes by the markers in their environment (spool,
watcher and job) and stops them once the job is over; a process started outside the scheduler carries no
marker. Find processes whose parent is 1 (`ps -eo pid,ppid,comm | awk '$2==1'`) and kill them by pid --
never by a pattern that also matches your own shell. `bench.py` refuses a device with more than 2 GiB in
use.

*Caught by:* `processes_with_marker`.

### Every probe on one node looks 3-4% worse at concurrency 1

Not the knob: the node. Identical configs differed by up to 4% between GB300 nodes of one cluster while
same-node repeats agreed within 0.1%. Judge against a same-host control (the campaign does).

## Python environment

### The import resolves to a different tree than the one you are measuring

`/root/.local` and `/root/.cache` are symlinked into a shared, relocated home
on this cluster, so another checkout's develop-install is on the path of every
container run. Export `PYTHONNOUSERSITE=1` at runtime. A worktree has the
same problem with a different cause; see `reason=mixed_checkout` above.

**But not during the native build:** the CUTLASS kernel CMake runs
`setup_library.py develop --user` and then imports `cutlass_library`, so the
user site must be *enabled* for that step. Isolate the `pip install -e` step
and everything after it, not the build itself.

### The gate measures the wrong build entirely

`environment.python` is the whole answer: point it at the editable install you
intend to measure. Resolving the interpreter once, in the spec, is what keeps a
run from silently gating a different build — and it is why no agent ever
constructs a launcher command.

---

## The agent plane

### `You cannot dynamically configure MCP servers when an enterprise MCP config is present`

The presence of `/etc/claude-code/managed-mcp.json` — not its contents — makes
the SDK refuse every dynamically-registered server, including the in-process
one the typed-decision tools live in. The first symptom is otherwise an opaque
`ProcessError: exit code 1` on the very first agent turn.

The workflow registers no MCP server of its own: each role reports by writing
`decisions/<unit>-<role>-<turn>.yaml`. If you see this error, it comes from an
`extra_mcp_servers` entry in the `agents` block, which such a host refuses.

### `API Error: EROFS: read-only file system, mkdir '.../credentials.lock'`

Reported as `stop_reason='stop_sequence'` with **zero tokens**, so it reads as
a model refusal. It is not: no agent ever started. The spawned CLI's first act
is to take a lock beside its credential store, and a sandboxed `$HOME` is
read-only.

*The answer is `--agent-transport external`* (below): an already-authenticated
session runs the turns, so no child process needs a writable credential store.

### `Not logged in · Please run /login` on a turn that used to work

This is the *second* half of the problem above. A credential store copied to
a writable home holds a **copy** of a refresh token the parent session also
holds, and a refresh rotates it — so one of the two copies loses. Measured on this cluster: authenticated cleanly at 12:12, failed at
12:25, byte-identical credentials. A previous run measured "about forty
minutes". Ten concurrent smiths make it worse, not better.

Copying harder does not fix a single-use token.

*The answer is `--agent-transport external`.* The turn becomes a file: the
orchestrator writes `<workspace>/dispatch/<id>.{system,prompt}.md` and blocks
until a driver writes `<id>.result.md` and then `<id>.done`. Any
already-authenticated session can be the driver. Everything the orchestrator
*decides* stays in Python — the stage machine, the queue walk, the dependency
batching, decision-file validation and every gate.

One thing to check if you write your own driver: **the role's system prompt
travels with the turn** (`<id>.system.md`). A driver that delivers only the
user prompt is running a different agent — no role, no invariants, no embargo
— and its answer still looks plausible.

*Caught by:* `probe_agents`, under the in-process transport: the probe's turn
fails with this text, and the run stops before its first planner turn.

### `... is older than the minimum version required by your organization`

Every in-process turn is refused at start-up, by the organization's policy, not
by the model. The Claude Agent SDK launches the `claude` it bundles, which only
moves when the SDK is upgraded, while the CLI you use by hand is newer. The text
reaches the CLI's stderr; the exception the turn raises may only say
`Command failed with exit code 1`.

Set `AGENT_FLOW_CLAUDE_CLI` to the `claude` that works in your shell
(`command -v claude`, `claude --version`) and start again. Without it a `claude`
on `PATH` newer than the bundled one is already preferred; the agent probe's
failure names the binary and version the layer launched.

*Caught by:* `probe_agents`, which reports the CLI's stderr with the failure.

### `connect EPERM ... srt-mux-*.sock`, in a turn that ended normally

The turn ran, but its Bash tool could not start its own sandbox, which the
managed policy requires: the orchestrator was launched from inside another
agent's session (its sandbox; `SANDBOX_RUNTIME` or `CLAUDECODE` is set in the
environment), and a sandbox cannot be started from inside one. The model
reported that Bash failed and ended its turn, which reads as a turn that worked
and decided nothing.

Launch the orchestrator from a plain terminal or a tmux session on the login
node, or use `--agent-transport external` and let the session you are in drive
the turns.

*Caught by:* `read_probe_file`: the probe's file holds the nonce's SHA-256,
which only a shell that ran writes, so a turn that only *says* it ran fails.

### `agent probe: N of M agent(s) cannot run a turn with Bash here`

Preflight's agent probe failed; the run stopped before its first planner turn.
Under it, per agent: what failed, what the turn said, the CLI's stderr, and its
files (`gates/agent-probe/<n>/<id>.*`: the prompt, the activity log, the
answer). Then what to do, from that text: the two entries above, a login, or
`--agent-transport external`. `the turn's shell ran, but git -C ... rev-parse
HEAD printed ...` means the turn cannot read the checkout it is to change
(another mount namespace, or git's `dubious ownership`). Set
`MV2FLOW_SKIP_AGENT_PROBE=1` only when the probe itself is what is broken.

*Caught by:* `probe_agents`.

### `<id>: no answer within Ns (the <role> turn timeout); the turn was stopped`

An in-process turn ran past its role's bound (README, `--agent-turn-timeout`).
It was cancelled, which stops its CLI, and raised `TurnTimeout`, as an external
turn nobody answered does: a task's turn is retried once, then the task fails
to the planner. What it did up to then is in the activity log the message names.

*Caught by:* `InProcessTurnExecutor`.

---

## Filesystem and git

### Everything stops at once with unrelated-looking errors

Check `TMPDIR`. Engine builds, NVRTC scratch and harness output all default
there, and on a shared host another tenant can fill `/tmp` to zero mid-run.

*Caught by:* `check_tmpdir`, which rejects a `TMPDIR` under `/tmp` — reading
the value **declared in `task.yaml`**, deliberately not the orchestrator's own
`os.environ`, because a run whose scratch location is implicit is one nobody
can reproduce.

### `ipc path "..." is longer than 107 characters`

Followed, from the same path, by `zmq.error.ZMQError: File name too long`.

107 is `sizeof(sockaddr_un.sun_path)` — a kernel cap on unix domain socket
paths, not a ZeroMQ or TensorRT-LLM choice. The LLM API's RPC binds an IPC
socket under `TMPDIR` named `rpc_test_<uuid4>`, a fixed 45 characters plus a
separator, so **`TMPDIR` must be at most 61**.

This is about the length of the *name*, not the filesystem. Unix sockets bind
fine on Lustre here — tested on `/tmp`, `/dev/shm` and both Lustre spellings,
all four succeed. Anyone copying a `TMPDIR: /tmp` workaround should know it
works because `/tmp` is *short*, not because it is local; point `TMPDIR` at a
shorter path on the same storage instead.

On a cluster whose storage paths are deep this is the property that breaks
first: a project scratch directory of the form
`/scratch/<fs>/portfolios/<org>/projects/<project>/users/<user>/` was 78
characters before the run appended anything.

*Caught by:* `check_tmpdir`, which derives the bound from
`UNIX_SOCKET_PATH_MAX` and `RPC_SOCKET_BASENAME_LEN` and puts the arithmetic
in the message. Seconds in preflight instead of a four-rank engine load and a
ZeroMQ traceback that names neither the spec nor the workflow.

### `Unable to create '.git/index.lock': File exists`

Two agents committing in the same instant, or a stale lock from a crashed
process. Before removing anything: confirm no git process is running
(`pgrep -a git`) and check the lock's age and size — a zero-byte lock with no
owning process is stale.

The workflow removes the *cause* rather than the lock: smiths commit through
`bin/commit.sh`, which takes an `flock`, so concurrency becomes a short queue.
Exit code **75** from that script means "waited for the lock and did not get
it" — a queue, not a broken repo.

### `commit.sh: REFUSED: ...` (exit 2) and `REFUSING TO LEAVE THIS COMMIT AS IS` (exit 1)

Exit 2 is a refusal before anything was staged; fix the call, not the repo.
The refusals, each of which once reached a branch:

- `is a symlink git does not track` — a worktree's link to the main
  checkout's native build (`tensorrt_llm/bindings` and friends). `.gitignore`
  names the directory form only, so the link is untracked and visible, and a
  `git add -A` committed it. Name real files only; leave the link alone.
- `sandbox placeholder` — `.mcp.json`, or anything under `.claude/` or
  `.cc-writes/`, which the operator's sandbox leaves wherever its shell goes.
- `outside the paths this run may commit` — `commit_policy.allowed_paths` in
  `task.yaml` (default `tensorrt_llm/ tests/ cpp/ examples/`).
- `Co-authored-by` / `Signed-off-by` in the message — the script signs off
  as the repository's user itself, and the repository takes no co-authors.
- `no commit identity` — the repository has no `user.name`/`user.email`.
  Set them (or export both `MV2FLOW_GIT_NAME` and `MV2FLOW_GIT_EMAIL`);
  the CLI refuses to start without one for the same reason.

Exit 1 with `REFUSING` means the commit was made and undone: the file list
differed from the request, a symlink was added, a sign-off did not carry the
author's email, or a `Co-authored-by` appeared — usually a hook or an
environment variable, never the agent. Delivery runs the same checks over
every commit since the merge base (`git:commit-hygiene`); a commit missing a
sign-off has to be re-signed as its own author, not as whoever runs the
rebase.

### `could not lock config file .git/config: File exists` and the lock will not delete

Seen on Lustre with `Device or resource busy`. Do not fight it: pass the
identity on the command instead of writing it to the config —

```bash
git -c user.name="..." -c user.email="..." commit -s ...
```

### A suite reports every packaged skill as malformed

A Claude Code session writes an empty `.claude/.cc-writes/` into whatever tree
it touches, and a filesystem walk treats it as a skill directory. git does not
track it, so `git status` shows nothing — which is the most confusing possible
way for a suite to go red.

*Caught by:* `_skill_dirs()` in `test_skills.py`, which skips dot-directories.
Apply the same rule to any new walk over a packaged tree.

---

## Measurement traps

These produce **numbers**, not errors, which makes them the most expensive
category here. A wrong number is believed.

### The accuracy gate reports a large regression that is not there

Measured once: **63.50 against a 95.53 anchor, a 27-point phantom.** The anchor
had been scored with a chat template and the gate had not. A benchmark's
*protocol* is part of its identity — a reasoning model scored as 5-shot
completion lands near the floor whatever its ability.

Declare it: `accuracy.benchmark_args: [--apply_chat_template]`.

### A feature's gate re-measures the base path

The `mtp` unit's accuracy gate must carry a `speculative_config`. Without one
it measures the base model and reports the number as the speculator's. Use
per-unit `accuracy.llm_api_options`:

```yaml
accuracy:
  llm_api_options:
    core: /path/to/oracle_llm_api.yaml
    mtp: /path/to/mtp_llm_api.yaml
```

### A gate measured under `auto` is not measuring modeling_v2

Under `modeling_v2=auto`, a configuration that misses a target's routing
criteria silently gets the built-in implementation — and the number reads as
modeling_v2's. Anything you will attribute to modeling_v2 runs under
`require`, which raises instead of falling back.

It must be exported **before the ranks start**, not merely before `LLM(...)`:
workers receive the environment as it stood at MPI init, so a driver resolving
a modeling_v2 target while its workers resolve the built-in is exactly the
silent split the switch exists to prevent.

### `the engine did not run the declared configuration -- <field>: declared X, engine Y (read from ...)`

The gate built its engine as the unit declares, and the engine ran something
else. Each field names where its value was read: `worker` is rank 0's
executor after it loaded the model, `llm.args` is the validated LLM-API
arguments, and `engine log` is a rank's own warning. Measured once: block
reuse off in every quickstart gate of a hybrid model for about 30 hours while
each gate passed. The fix is in the declaration, not the target:

- `block_reuse: declared on, engine off` — for a hybrid model, set
  `kv_cache_config.mamba_state_config` in the unit's options file. Otherwise
  the engine had its own reason (an MLA configuration, an encoder-decoder
  model); find its warning in the log, and turn `kv_cache_reuse` off for the
  unit if the engine is right.
- `cuda_graph: declared 19 batches ..., engine 34 batches ...` — two launch
  paths resolve different lists. Run `--dry-run`: preflight's
  `check_capture_lists` names the unit.
- `kv_cache_dtype: ... engine asked for nvfp4 and resolved FP8` — the worker
  fell back (NVFP4 KV on SM107, for one). The run cannot serve the declared
  dtype on this device.
- `modeling_v2: declared off, engine require` — the tree's quickstart passes
  its own `modeling_v2`, and the harness adds the gate's only when it is
  absent (`engine_kwargs` in `checks/harness/engine_driver.py`).

A field the engine record does not carry shows as `NOT ATTESTED <field>` in the
`[engine ...]` part of the summary. It is never read as agreement. Where rank
0's worker runs in another process (every member under `trtllm-llmapi-launch`,
or a TP1 engine without `TLLM_WORKER_USE_SINGLE_PROCESS=1`), block reuse is
read from what the engine reused instead (`read from reuse probe`, `read from
cached tokens`), and stays `NOT ATTESTED` only when the run sent no probe and
no request resumed.

*Caught by:* `attest_engine` (`checks/gates.py`), over the harness's
`MV2FLOW_ENGINE` and `MV2FLOW_REUSE` records (`checks/harness/engine_facts.py`,
`checks/harness/engine_reuse.py`).

### `the unit asks for KV-cache block reuse and the engine reused nothing: ...`

The unit asks for block reuse, and the engine resumed no request from its KV
cache although it had a prefix to resume from. Either the reuse probe's second
prompt (the same prompt again, sent after the first finished) was computed from
token 0, or more requests than the engine runs at once shared a long enough
prefix with an earlier request and none resumed. The configuration can say
reuse is on and still reuse nothing. Measured once: the accuracy logs showed
reuse on for a hybrid model, and nothing in them showed a single hit, so
"block reuse works" rested on gsm8k scores alone. At every multi-rank member,
block reuse was not attested at all.

- A hybrid model resumes only where it holds a recurrent-state snapshot.
  Check `kv_cache_config.mamba_state_config` in the unit's options file. The
  probe is two KV blocks longer than the first snapshot point, so a policy
  whose points the probe never reaches is not the cause; a policy the engine
  did not run is (`mamba_snapshot` in the same `[engine ...]` segment).
- A speculator whose `context_recompute_tail` is `-1` recomputes every
  resumed prompt from the start, and the responses then say 0. Turn
  `kv_cache_reuse` off for that unit if this is intended.
- `reuse probe not read (...)` in the summary means the probe did not run
  (`skipped`: it does not fit `max_seq_len`; or the error it raised). Nothing
  is judged from it then.

Each `MV2FLOW_RECORD` of the engine driver also carries the request's own
`cached_tokens`, so the workload's requests can be checked one by one.

*Caught by:* `reuse_shortfall` (`checks/gates.py`) in the boot gate, both
acceptance legs, both envelope legs and the accuracy gate, over the
`MV2FLOW_REUSE` record.

### `[WARNING: LOW MARGIN: 0.52 GiB free on device 0 ... once the engine was up]`

The gate passed with little device memory left once its pools were sized and
its CUDA graphs captured. The next change that adds memory to that
configuration fails at model creation in a unit nobody touched. Measured once:
MTP depth 7 at TP1 ran with 0.515 GiB free. When the vision tower was
integrated (+3.26 GiB per rank), that configuration ran out of memory, and the
failure surfaced 7.5 hours later. Recheck the configuration with the least free
memory first after an integration, and find what holds memory twice (the
diagnosis that time: a target kept views of the estimation pool's buffers).
The threshold is the larger of `runtime.min_free_memory_gib` (default 2) and
`runtime.min_free_memory_fraction` (default 0.02) of the device.

*Caught by:* `MemoryMargin` (`checks/gates.py`). It warns; the verdict is not
changed.

### `the overlap scheduler changed the greedy output at batch 1`

Two causes produce this failure. The target may read generation-step state
that was prepared before the previous step's acceptance was known. Or each
leg's process made its own kernel choice. Each leg is a separate process, and
a `@triton.autotune` kernel picks its config by timing, per process. Two
configs reduce in a different order, so one configuration can produce two
outputs from one process to the next. Measured once: two output modes of one
configuration, matching the `num_warps` that one FLA chunk kernel's autotune
picked in each process. One prompt first differed at token 0, which comes
from the context step, and a scheduler defect shows in generation steps.

Read the rest of the verdict before the target:

- `the overlap leg, rerun in a fresh process, reproduced itself` — the
  difference follows the configuration. Look for a stale read.
- `did NOT reproduce itself` — the comparison cannot judge the scheduler.
  Something a process picks for itself is still unpinned. Run the leg with
  `TRITON_PRINT_AUTOTUNING=1` twice and compare the configs it prints.
- `no Triton autotuning was cached` beside a target with autotuned Triton
  kernels — this Triton ignores `TRITON_CACHE_AUTOTUNING`, so each leg tuned
  its own.

Do not set `TRITON_CACHE_AUTOTUNING` and `TRITON_CACHE_DIR` in
`environment.env` to get around it. That changes every gate, not only the
compared legs, and the gate's own pins override those variables anyway.

*Caught by:* `pin_tuning`, which gives every leg of one gate run the same
AutoTuner and Triton autotuning caches, emptied at the start of the run.

### `INSENSITIVE, not a pass: ... with the received state destroyed (...) continuation #N (W words) was still token-identical`

The disagg witness found both legs token-identical, then ran leg B again on
its longest prompt with `MV2FLOW_WITNESS_ABLATE=state`, and the output did
not change although the state the generation engine received was destroyed.
A witness whose workload does not read that state passes a hop that drops it.
Measured once: with a hybrid model's recurrent state and conv window zeroed
after the hop, prompts of `one two ... ten` at 131, 2050 and 4097 tokens
stayed identical, while prose prompts of 374 and 3106 tokens diverged at
tokens 1 and 11. The attention pages alone continue a counting pattern.

- A hand-written prompt that repeats itself or is a famous text: replace it
  with prose the model has to read. `prompt_tokens` generates prose
  (`checks/witness_text.py`), but past the passage's length
  (`NATURAL_TEXT_WORDS`, about 5400 words) it repeats the passage; for a
  longer prompt, declare prose of your own in `prompts_file`.
- Leg B printed `Witness ablated:` but destroyed nothing that matters: check
  that its hook zeroes the slot the request was given
  (`_get_mamba_slot_for_request`) on the generation engine, on every rank,
  after the transfer completed and before the first decode step.

*Caught by:* the self-test in `run_equivalence_witness` (`checks/witness.py`).

### ``the self-test could not run: the self-test leg (MV2FLOW_WITNESS_ABLATE=state, ...) printed no `Witness ablated:` line``

- The block names a `contrast_command` that does not implement
  `MV2FLOW_WITNESS_ABLATE`. It is the run's code, outside this package, and
  the README's witness workloads section gives its contract: zero, on the
  generation engine, every layer's recurrent state (else the KV pages) of each
  generation-only request once its transfer completes, and print
  `Witness ablated: <what>` per request. Under a multi-rank engine the hook
  must reach every rank's process, not only the one the launcher talks to:
  put it in a `sitecustomize` on the engines' `PYTHONPATH`. Or drop the
  `contrast_command` and let the flow's own leg B run.
- The flow's own leg B (`checks/harness/disagg_leg_b.py`) installed its hook
  but the executor never took a request up through
  `PyExecutor._prepare_disagg_gen_transmission_complete`: the tree under test
  hands a received request to generation somewhere else. Find where it does,
  and move the hook there.
- A line `Witness ablation refused: the hook could not be installed: ...`
  above says why the generation engine's `sitecustomize` failed.

### `Witness ablation refused: leg B cannot reach the state a request received on this engine's caches (...)`

The flow's own leg B was asked to destroy what the generation engine
received, and none of the engine's cache managers exposes it: no manager has
recurrent layers (`mamba_layer_offsets`) with `get_conv_states` and
`get_ssm_states`, and the KV-cache manager does not name a request's pages
(`layer_offsets`, `get_batch_cache_indices`, `get_buffers`). The generation
engine is refused when it is built, so the self-test leg does not complete
and the witness does not pass; it is never skipped. Teach `resolve_ablation`
the manager named in the parentheses, or name a `contrast_command` that can
reach it.

### `leg B starts a context and a generation engine of N rank(s) each, so it needs 2N devices and sees M`

The flow's own leg B puts the context engine on the first half of the devices
it can see (`CUDA_VISIBLE_DEVICES`, else every GPU on the node) and the
generation engine on the second. A member wider than half the node has no
disaggregated leg on it. Run the witness at a member at most half the node's
width, or give leg B a node with twice the member's devices.

### `the context engine exited N before it was up` / `did not come up before leg B's deadline`

One of the flow's leg-B engines died while it was being built, or never
connected back. Its own output is above the line. Check, in order: the
`cache_transceiver_config` (both engines run the same one; `DEFAULT` resolves
to NIXL unless an environment variable selects another backend, and the
Python transceiver runtime drives only NIXL), the interpreter
`MV2FLOW_WITNESS_ENGINE_LAUNCH` names, and the devices each engine got.

### `leg B did not report pinning each hop to its attention-DP rank pair (MV2FLOW_WITNESS_RANK_PAIRS=...)`

At a `dep` member, leg B must pin prompt `i`'s context-only and
generation-only requests to the `i`-th pair, with
`SchedulingParams(attention_dp_rank=..., attention_dp_relax=False)`, and print
`[i] Witness ranks: ctx=<c> gen=<g>`. Without the pin, the router breaks ties
by rank, and a leg that sends one request at a time crosses every hop from
context rank 0 to generation rank 0, so a transfer defect on any other rank is
never compared. `reported ctx=0 gen=0 for 0:1` means the launcher printed the
line without pinning.

### `llm.get_stats()` returns nothing

The feature witnesses for overlap scheduler, chunked prefill and block reuse
are **unverified for this reason**, not because the features are absent. A
self-diagnosing witness reported `[stats] 0 iterations`. Earlier attempts also
failed for unrelated reasons — concurrent `MPI_Init`, `TLLM_DISABLE_MPI=1`
routing to a missing Ray, and wrong stats field names returning all zeros —
so do not assume the stats API is the only thing between you and a witness.

*Open.* If you need these features proven, this is the thing to fix first.

### The survey says good catalog entries are "Rotted — do not spend these"

```
**Rotted** — the entry names an `impl` that does not resolve in this build.
- `comm/allgather` → `torch.ops.trtllm.allgather`
- `quantization/fp4_quantize` → `torch.ops.trtllm.fp4_quantize`
- `attention/fused_qk_norm_rope` → ...
```

Check one by hand before believing it. **`torch.ops.<ns>` is a lazy proxy**:
`dir()` lists only the names something has already resolved, not the names that
are registered. Measured on a GB300 build:

```python
len([n for n in dir(torch.ops.trtllm) if not n.startswith("_")])   #  91
len({s.name for s in torch._C._jit_get_all_schemas()
     if s.name.startswith("trtllm::")})                            # 273
```

With the short answer, 13 of 15 "rotted" entries resolved fine — and they were
exactly the vocabulary a quantized hybrid MoE target needs, so the plan would
have routed around all of them and dispatched a smith to re-onboard each one.

*Caught by:* `survey.py`, which enumerates from the schema registry and
records `op_enumeration: schema_registry | dir_fallback` so a reader can tell
which measurement they are looking at.

### A feature unit's gate fails to start with `no such option`

The evaluation **protocol** is options of the *benchmark subcommand*, and the
subcommands do not share an option set — `gsm8k` takes `--apply_chat_template`
and `--fewshot_as_multiturn`, `mmmu` takes neither. A run-wide
`accuracy.benchmark_args` therefore does not make a multimodal unit's gate
imprecise; it makes it fail to start, with a click usage error that reads as a
spec mistake.

`benchmark`, `reference_yaml`, `llm_api_options`, `discriminators` and
`benchmark_args` all take the per-unit mapping form. Use it.

### Two `task.yaml` copies disagree

The spec is copied into the workspace, and on a resume the two can drift.
`_refresh_workspace_spec` re-copies it, but if you edit a spec mid-run, edit
the source and resume — do not edit the workspace copy.

### `it ran in another environment`, `opaque: ...` or `environment not checked` from `tests affected` / `tests compare`

The test cache reuses a result only where nothing the test ran on changed,
and it says why it did not, per test:

- `it ran in another environment`: the job's interpreter, installed
  packages, GPU models or count, or a `TLLM_`/`TORCH_`/`NCCL_`/`CUDA_`-style
  setting differs from the recording's. Run the tests in the same image on as
  many devices each time; a 1-GPU job skips multi-GPU tests that a 4-GPU job
  runs.
- `opaque: imported \`tensorrt_llm\` from outside the tree`: the tests
  imported an installed copy instead of the checkout, so nothing they ran is
  the tree's. Run from the checkout's root, as the printed command does.
- `opaque: starts ...` or `hands work to other processes`: from that test on
  its process runs code the recorder cannot see, so its results are reused
  only at the commit they ran on.
- `its commit is not in this repository`: `--repo` is a clone without the
  recorded commit; name the checkout the run integrates into.
- `environment not checked`: nothing was recorded at the head, so the
  reused results were never compared with the head's environment. Run the
  printed driver command without a selection; it records only the
  environment.

*Caught by:* `head_environment` (the environment check); the rest is the
cache declining to guess.

---

## When none of this fits

Say so rather than working around it. The agent prompts carry this instruction
verbatim — *"If this prefix does not fit what you need to run, that is a
**symptom to report in your final message** — not a problem to solve"* — and
the same applies to a human operator. A run that spends its budget
rediscovering how to submit a job is the failure mode this whole workflow was
shaped to avoid.
