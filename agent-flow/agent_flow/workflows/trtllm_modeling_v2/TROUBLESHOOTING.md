# Troubleshooting — infrastructure, not modeling

Every entry here cost a real run hours, and every one **read as a model or
target defect while it was happening**: an engine that will not boot, a gate
that reports a 27-point regression, a test that says nine skills are
malformed. None of those sentences mention the environment, and all of them
were the environment. So the first question when something fails is **"has
this host ever done this successfully?"**

Entries are keyed on **verbatim output**, the string you actually saw: search
this file for the error text, not for a description of it. Each entry says
whether the workflow already catches it (`*Caught by:*`); reaching a symptom
a check should have caught is itself worth reporting.

## Preflight catches most of this before an agent exists

Run `--dry-run` first: it resolves the environment and stops before the first
agent turn.

| check | what it proves |
|---|---|
| `check_checkpoint` | the checkpoint parses and declares an architecture |
| `check_tmpdir` | `TMPDIR`/`CLAUDE_CODE_TMPDIR` are set and not under `/tmp` |
| `check_host_tools` | notes, never a stop: a git older than 2.40 when perf is on (the tune campaign then stacks each winner as one change), and on a `local` run a `python3` on `PATH` older than 3.7, which the native rebuild script refuses |
| `claim_devices` | a local run: enough idle devices of the declared arch, by free memory and a lone-process probe. Under a scheduler the scheduler hands out devices |
| `check_multirank_substrate` | NCCL collective + MPI spawn answer under a deadline, for a multi-rank run |
| `probe_agents` | in-process turns: every agent the roles run on takes a turn here and its **Bash tool runs a command**, judged by the file it writes |
| `substrate_gate` | **this host generates a token** on the stock path under this run's exact runtime block |

`substrate_gate` *generates* rather than constructs, because the NVRTC failure
below appeared on the first generated token, after the engine had loaded 22 GB
of weights, allocated the KV cache and finished warm-up. A resumed run does
not run it again when its last pass, `<workspace>/gates/substrate/pass.json`,
is under 12 h old and was recorded for the same resolved environment,
checkpoint, runtime block, options file and checkout path; a repair that
record cannot see (an image re-imported under the same name, a package
installed into the same interpreter) needs that file deleted before the
restart. The agent probe's pass, `<workspace>/gates/agent-probe/pass.json`, is
reused the same way.

---

## Launch substrate

### `MPI_ERR_SPAWN: could not spawn n processes`

**Not a model problem.** The container runtime forbids `MPI_Comm_spawn`, which
the default LLM API path uses; under pyxis/enroot that is normal. The decision
is in `llmapi/mpi_session.py`:

```python
def need_spawn_mpi_workers(model_world_size: int) -> bool:
    return get_mpi_world_size() == 1 and model_world_size > 1
```

The fix is to make the ranks exist before the engine starts, which is what
`trtllm-llmapi-launch` does: one task per rank, rank 0 runs the leader and the
real task, the rest run workers. The workflow emits it at world size > 1; an
allocation sized without a matching `--ntasks` is the whole bug.
`TLLM_WORKER_USE_SINGLE_PROCESS=1` is not the general fix: the function behind
it is `enable_worker_single_process_for_tp1`, a workaround for TP1 that does
not scale past one rank. At world size 1, use no wrapper and clear the MPI
environment before exec: mpi4py initializes MPI whenever PMI variables are
present, and inside a PMIx-managed step they always are.

*Caught by:* `substrate_gate`. `check_multirank_substrate` does **not** catch
it: its probe uses `torchrun`, a different launcher, so it goes green first.

### The engine hangs at startup and never returns

A wrong launch substrate **hangs rather than raises**, usually an MPI
connect-back on an interface that does not carry it. Without a deadline this
surfaces hours in as "the engine never booted".

*Caught by:* `check_multirank_substrate`, which fails on its deadline.

### `There are not enough slots available in the system`

In full: *"There are not enough slots available in the system to satisfy the 4
slots that were requested by the application: trtllm-llmapi-launch"*, before
any Python runs. Open MPI takes its slot count from the resource manager, and a
`session_spool` job is submitted as **one task** that owns the node, so `-n 4`
asks for four slots against one. Do **not** raise the session's
`ntasks_per_node`: that starts four copies of the spool watcher, which then
race for every piece of work.

*Caught by:* `_mpirun_launch`, which adds `--oversubscribe`.

### Four `mgmn_leader_node` processes, and a log that stops growing

No error and no exit: every rank thinks it is rank 0. `trtllm-llmapi-launch`
derives the rank from `SLURM_PROCID` before `OMPI_COMM_WORLD_RANK`, and reads
`SLURM_NTASKS` before `OMPI_COMM_WORLD_SIZE`. Under `mpirun` inside a Slurm
allocation `SLURM_PROCID` is 0 for every rank and `SLURM_NTASKS` is 1
(measured: `SLURM_PROCID` read `0, 0, 0, 0` where `OMPI_COMM_WORLD_RANK` read
`0, 1, 2, 3`). MPI itself is fine.

*Caught by:* `_mpirun_launch`, which prefixes `env -u SLURM_PROCID -u
SLURM_NTASKS ...` so the launcher falls through to the Open MPI variables.

### `UCX ERROR uct_iface_open(tcp/rdma_vf_rail0) failed`

Noise, on every rank: UCX fails to open one IPv6 RDMA virtual-function
interface, falls back, and the run completes with correct output.
`UCX_TLS=sm,self` silences it and fixes nothing. If the run is also hanging,
the cause is the entry above.

*Not caught, deliberately.*

### Two jobs die together, both at `MPI_Init`

Concurrent `MPI_Init` in one allocation killed two compute sessions in one
run. The spool scheduler removes the allocation's MPI context from every job,
which addresses the likely mechanism; if it recurs, serialize the jobs that
initialize MPI (`--gpus all`) or give them distinct allocations.

*Not caught.* The mechanism was not isolated further.

### Setting `TLLM_DISABLE_MPI=1` makes it worse

It routes the executor to Ray, which was not installed, so the failure moves.
For the spawn error above, `TLLM_WORKER_USE_SINGLE_PROCESS=1` is the setting
that works, at world size 1.

---

## The engine boots but will not run

### `Impossible to fit in any sequence in kvCache`

The engine inherited the checkpoint's `max_position_embeddings` (262144 is
common) and sized the KV cache for it. Declare the runtime in `task.yaml`:

```yaml
runtime:
  extra_flags:
    - --max_seq_len=8192
    - --max_batch_size=8
    - --kv_cache_fraction=0.6
```

*Caught by:* `substrate_gate`, which runs under the same `runtime` block the
gates use.

### `NVRTC ... could not open source file "cuda.h"`

Two independent causes:

1. **The venv's `bin` is missing from `PATH`.** NVRTC finds the toolkit headers
   relative to the interpreter's environment. Check this first.
2. **The generation FMHA kernel takes the NVRTC path** unless the KV dtype is
   `E2m1` (`kernels/trtllmGenKernels/fmha/fmhaKernels.h`; grep for
   `options.mDtypeKv != tg::Dtype::E2m1`). On a build whose NVRTC has no header
   search path, that path dies compiling; `--kv_cache_dtype fp8` takes the
   other branch.

Rebuilding with `--nvrtc_dynamic_linking` does **not** fix it, and "fp8 fixed
it" works for the second reason, not the first. Check `PATH` before you
rebuild anything.

*Caught by:* `substrate_gate`, because it generates a token.

### The recurrent state runs at the wrong precision, silently

A checkpoint declaring `mamba_ssm_dtype: float32` still gets the engine flag's
default of bfloat16. Pass `--mamba_ssm_cache_dtype=float32` in
`runtime.extra_flags`.

*Not caught.* The surface survey lists it among the checkpoint's demands at
plan time.

---

## Cluster access

### `sinfo` / `sbatch` / `squeue` cannot contact the controller

On some clusters the Slurm CLI is unreachable from the session and an MCP
broker is the only path; the cluster is not down. Use `environment.kind:
session_spool`: the driver session starts each compute session through the
broker (`trtllm-modeling-v2 session submission` prints the call), and
everything after that is files.

### A broker job "fails" after exactly 300 seconds but is still running

The MCP layer aborts a tool call after 300 s of silence, so the blocking
`slurm_run` reports failure for a healthy job and leaves it running. Use
`slurm_submit`, poll `slurm_job_status`, then read the log file.

### Work started in a job step dies when the step ends

Work started inside a `slurm_exec_in_job` step does not survive the step's
teardown, `nohup` included (measured). With the 300 s cap and a 26 GB image
costing 4-8 minutes to import per job, this is why `session_spool` exists: one
allocation whose own main process is the spool scheduler, and work is a file
(`pending/<name>.sh` in, `done/<name>.rc` and `logs/<name>.log` out).

### `invalid_mount ... world-writable governing directory`

```
{"error":"invalid_mount","status":400,
 "detail":"mount source has a world-writable governing directory
           /.../artifacts (post-validation component-swap vector)"}
```

A directory **above** the path you asked for is world-writable (CI artifact
trees usually are), so a checkpoint published there cannot be mounted where it
lives. Stage it into a directory you own: inside the checkout, excluded via
`.git/info/exclude`, since the orchestrator's sandbox grants write access only
to its declared working directories (a copy elsewhere fails with `Read-only
file system`, which reads as a quota problem).

### A job submits fine by hand but the workflow's submission is rejected

Check the GPU request form: NVL72 policy rejects job-scoped `--gpus`, and
`gpus_per_node` is the only form that submits there
(`environment.gpus_per_node`).

---

## Compute sessions and the spool

A session is an allocation whose main process is the spool scheduler
(`infra/substrate/scheduler.py`). Every job's verdict is in
`done/<name>.json`, whose `reason` names what happened; `spoolctl.py status`
shows what runs, what waits and why.

### The safe-kill rule, and a whole allocation that died when one job did

Six session deaths in one run had one cause: a process that inherited the
allocation's Slurm/PMI context died abnormally (a crash, a timeout, an
operator's `kill`), and Slurm's MPI error handler (`_errhandler status=-61`)
killed the **whole allocation**, scheduler, orchestrator and every queued job.

**The safe-kill rule:** never signal a process that carries the allocation's
MPI context (`SLURM_*`, `PMI_FD`, a `slurm.pmix.*` `PMIX_NAMESPACE`). The
scheduler removes that context from every job it starts, and still checks each
process before signalling it: one that carries the context is left alone, its
job marked failed and its devices quarantined until it exits. To stop a job,
use `spoolctl.py cancel <name>`. Never `kill` a job's processes by hand, never
`pkill`/`killall` by pattern, and never kill the scheduler: it is the
allocation.

*Caught by:* `scrub_environment` at launch and `is_mpi_attached` before every
signal. `trtllm-modeling-v2 session watchdog` reports quarantined devices.

### `no watcher heartbeat in <spool>: the session is not running` (exit 70)

Either the session is down, or it watches a different directory: a
hand-edited entry script copied from another run came up on that run's spool.
Compare `spool=` in the spool's `watcher.status` with the path you submit to,
and render entry scripts (`trtllm-modeling-v2 session render`).

*Caught by:* `queue_job`, which refuses a spool with no live heartbeat.

### A job waits while devices sit idle

Read the reason in `spoolctl.py status`:

| reason | meaning |
|---|---|
| `waiting for the whole node to drain` | a whole-node job is next; slot jobs are not admitted until it has run |
| `queued behind <job>` | a GPU job ranked above it is waiting for devices it will need |
| `host memory N GiB available, M needed` | it declared `host_mem_gb` (whole-node jobs inherit `engine_host_mem_gb`); the previous engine's host memory is still being released |
| `device memory in use: <dev> (N MiB) > M MiB` | a device it would get still holds memory no running job accounts for; after `device_release_wait` the device is quarantined (`quarantined, memory not released: ...`) and the job is placed without it |
| `native rebuild of <tree> in progress` | a rebuild was announced on this session, of the job's tree or (`<job tree> shares its native build`) of the checkout its snapshot or worktree links to |
| `... waits for the successor session` | the allocation ends before the job's wall limit would |

*Caught by:* `_admit`, which writes the reason into the status file.

### A KV-cache pool or slot count smaller than the same engine's elsewhere

An engine sizes its pools from the device memory free at boot. Started beside
memory a cancelled job had not given back, an mtp2 tp1 boot reserved 642
cache slots where the identical flags reserve 1237, and its gate certified the
smaller pool. The scheduler holds a GPU job while a device it would get has
more than `device_max_used_mib` (1024 MiB) in use, and quarantines a device
still over it after `device_release_wait` (5 minutes). If you see the symptom
anyway, check that `nvidia-smi` runs in the session's container: without it
nothing waits.

*Caught by:* `_place`.

### `reason=wall_limit` (rc 124)

The job ran past `--wall`, counted from its start; time queued is not charged.
A gate's timeout is its wall limit.

### `reason=stalled` (rc 210): `no progress for Ns after a failure line`

The log showed a failure (a traceback, `MPI_ABORT`, a segfault, a CUDA or NCCL
error) and then nothing for `failure_stall` (10 minutes for GPU jobs): a
rank-0 init failure with the other ranks waiting in a collective. The
receipt's `detail` holds the failure line.

*Caught by:* `_track_progress`.

### `reason=mixed_checkout` (rc 217)

The job's processes imported the engine package from more than one checkout,
or not from its `--tree`. An MPI worker imports from its working directory
ahead of `PYTHONPATH`, so pointing only `PYTHONPATH` at a worktree runs it on
rank 0 and the main checkout everywhere else. Submit with `--tree <worktree>`,
which sets both. Every process's import is in `logs/<name>.resolve.jsonl` and
the log's `[mv2flow-resolve]` lines.

*Caught by:* `_judge_provenance`.

### `reason=memory_limit` (rc 211)

The job's resident memory exceeded its own `--mem-gb`.

### `reason=host_memory` (rc 214)

The memory the job declared never came back within `host_mem_wait`. The
receipt names the largest jobs on the session; if none holds it, a process
outside the scheduler does.

### `reason=orphaned` (rc 212)

The scheduler that ran the job stopped heart-beating while the job's
processes still run on this host (`detail` names their pids), or no watcher
recorded starting it. A scheduler on the same spool closed it out rather than
run it again beside itself. It is not a verdict; once those processes are
gone, run it again.

### `reason=node_lost` (rc 218)

The allocation the job ran on is gone (requeued onto another node, ended or
failed) and the job was not run again: it asked not to be (`requeue=0`), it
had run again as often as it may, or its cancel was requested; `detail` says
which, and the receipt's `attempts` lists every allocation it ran on. It is
not a verdict; run it again.

*Caught by:* `_lost`.

### `spoolctl: <name>: ...; queued again for run 2 of at most 2`

The allocation running the job was lost, and the next scheduler on the spool
queued the job again in its place. `spoolctl.py submit` and `wait` go on
waiting and return the new run's receipt; the cut-off run's log is under
`history/`. Do not resubmit it: that runs it twice. A job runs again once
unless its header says otherwise (`spoolctl.py submit --requeue N`;
`requeue=0` closes it out `node_lost`); the scheduler's `--requeue` sets the
session's default.

*Caught by:* `_recover_orphans`.

### `the session watching <spool> stopped heart-beating ...; <name> is still in running/` (exit 70)

`spoolctl.py submit` or `wait` stopped waiting because the session stopped
heart-beating, as it does while a requeued allocation comes back. The job is
not lost: the next scheduler runs it again or closes it out `node_lost`, and
`spoolctl.py wait <name>` returns that. `--session-wait 15m` waits that long
for a session instead.

*Caught by:* `left_in_spool`.

### `reason=cancelled` (rc 216), `reason=refused` (rc 213), `reason=launch_failed` (rc 215)

`cancelled`: someone ran `spoolctl.py cancel <name>`. `refused`: the job could
never run here (a header key the scheduler does not know, more GPUs than the
session hands out, a `--tree` the compute node cannot see); `detail` says
which. `launch_failed`: the scheduler could not start `bash` for it, a node
problem.

### `job name ... must start with this requester's prefix`

Each external turn gets a prefix (`r014` for request `014-kernel_writer`), and
the submit script refuses names outside it, so two turns cannot pick the same
name. `export MV2FLOW_JOB_PREFIX=<prefix>` and name jobs `<prefix>-<what>`.

*Caught by:* `check_name`.

### A process faulted (SIGSEGV) about a minute after someone rebuilt the native libraries

`build_wheel.py` rewrites `tensorrt_llm/libs/*.so` in place, under every
process that has them mapped. Rebuild only through the workflow's
`incremental_build.sh`, which gives every library a fresh inode first and
holds new GPU jobs against that checkout on every session until the build
ends.

*Caught by:* `refresh_inodes`.

### `... did not run to a verdict: ...`, and a run closed `session-lost:<stage>`

The gate's compute session stopped under it, so the gate measured nothing:
nothing is recorded and no assembler turn is spent. The orchestrator waits up
to `environment.session_wait` (default 1 h) for a session to heart-beat
again and re-runs the gate, at most three times in a row; past either bound
the run closes `session-lost:<stage>`. Bring a session back
(`trtllm-modeling-v2 session status`, `session allocate`) and re-run the same
command. A session that keeps coming back and dying is a substrate to repair.

*Caught by:* `_gate_until_verdict`.

### The allocation reached its time limit in the middle of a gate

Declare the session's `time_limit`. `handover_lead` before the end, the
scheduler writes `handover/request.json` and stops admitting jobs that could
not finish; `trtllm-modeling-v2 driver wait` returns `SESSION_REQUEST`, and
`trtllm-modeling-v2 session allocate` prints the call that starts the
successor on the same spool and the `session ack ... --after <job>` that
records it. A successor submitted and acknowledged before the request is
written already answers it (`handover/acks/<allocation>.json`).

*Caught by:* `_plan_handover`.

### `SESSION_REQUEST` for an allocation that ended a day ago

A request left behind by an earlier hand-over. A successor moves its
predecessor's request and acks to `handover/archive/` when it starts, and
`driver wait` archives a request whose watcher no longer heart-beats and says
so after its event:

```
TIMEOUT  nothing new
  note: session g3: archived request g3-1000027-1792258324 (allocation 1000027): allocation 1000027 has no live watcher: it ended
```

Nothing is asked of you. Do not archive requests by hand; `session requests`
names the ones it ignores.

*Caught by:* `stale_request` in `driver wait`; `_take_over_handover` when a
successor starts.

### `session allocate` prints `dependency="afterany:<job>"` for an allocation that has ended

A dependency on an ended job is met at once, and the session would run two
allocations. `session allocate` takes the job from the session's live watcher
(the one that ends last), never from the request on disk, and says which:

- `It starts when allocation <job> ends (...)`: submit it;
- `No dependency: no watcher on <spool> has heart-beat within 180s ...`: the
  session is down, and the call starts it as soon as Slurm schedules it;
- `Allocation <job> already has a successor acknowledged: job <id> ...`: one
  is queued; submit another only if that one is gone.

*Caught by:* `plan_successor`.

### `orchestrator: unknown (... writes no heartbeat (turns run in process) ...)`

`driver wait` found no fresh orchestrator heartbeat in `dispatch/`, so it read
the run's ledger (`inflight.json`) and the workspace lock: a lock held is a
running orchestrator from any host, a closed run says so in the ledger, and a
free lock means an exit only on the ledger's own host, because another host
may not see the lock. Run the operator loop (`driver wait --sessions-only`) on
the host the detail names.

*Caught by:* `_ledger_state`.

### The driver waits and nothing happens

`trtllm-modeling-v2 driver status` shows the orchestrator's state. `exited`
means a run that closed, crashed or lost its allocation (its log says which);
a relaunch adopts every unanswered turn and every answered turn of a task that
had no outcome yet (`ADOPTED`), so do not copy an old decision to a new
request's path. `CLAIMED ... OVERDUE` means a turn ran past its role's timeout
and its subagent has likely died: `release` it and dispatch it again.
`reload: draining (...)` means a reload waits for what it names; see the next
entry.

*Caught by:* `orchestrator_state`.

### A flow fix has to reach a running run, and a relaunch would cut work short

Do not kill the orchestrator to load it: a stop by hand cuts short a rebase,
gate jobs and a planner answer about to be read. Run `trtllm-modeling-v2
driver --workspace <ws> reload`: the orchestrator starts nothing new, waits
for what is in flight, checkpoints and re-executes itself on the code now on
disk ([design.md §9.5](docs/design.md#95-reload-and-drain)); worker turns go on
and are adopted. `driver wait` returns `RELOADED` once its loop runs again.

- `RELOAD_OVERDUE`: the drain passed its `--timeout` and names what it still
  waits for; wait, `driver reload --force` (a cut check runs again), or
  `driver reload --cancel`.
- `RELOAD_FAILED`: the re-exec failed (the reason is in `reload.json` and the
  journal); the run goes on in the old process, on the old code.
- `ORCHESTRATOR_EXITED` right after a reload: the new code did not start (an
  import error, say). Its log says why; fix it and relaunch the same command.
- The journal's `restart (hard-stop)` line names what a stop cut short, and
  `restarts.jsonl` keeps one line per start.

*Caught by:* `_reload_check`, which drains; `_record_restart` records what a
stop cost.

### `Agent stalled: no progress for 600s (stream watchdog did not recover)`

Too many concurrent turns on one account: seven of twelve died this way within
twelve minutes. `driver wait` hands out at most 8 worker turns at once
(`DEFAULT_MAX_INFLIGHT`, `--max-inflight`), whatever `--max-parallel-tasks`
(default 8) lets the orchestrator ask for; the planner's turn comes beside
that cap. If the stalls return, lower both.

---

## A task derived from the checkpoint (`--checkpoint`)

### `✘ intake: N block(s) in <workspace>/task.intake.yaml`

Phase 0 ran the request's checks before touching the host, and lists every
block (the journal has them under `intake checks failed`). Each is a fact
about the request, the checkpoint or the compiled spec: an open question the
derivation could not answer (`device.arch`, a witness's `contrast_env`),
members a checkpoint's heads or experts do not split across, an anchor row the
reference file lacks. Answer them in the request, or in the goal or
`site.yaml` with `--clean`, and re-run the same command.

`accuracy anchor [<unit>]: ... has no entry matching {'quant_algo': 'NVFP4'}`
keeps a quantized checkpoint off the unquantized row: add the row the
checkpoint's own format names, citing where the number came from, or set
`accuracy.discriminators` in the request.

### `the request changed after its run started from it`

The run is past preflight and `task.intake.yaml` no longer has the digest in
`task.intake.yaml.started`. Confirm the edit (`--confirm`, or
`trtllm-modeling-v2 intake confirm <request> --by NAME`) and re-run, or start
over with `--clean`. `changed after it was confirmed` is the same for a
confirmed request.

### `the workspace's request was derived for the goal ...` / `... from <checkpoint>`

The workspace already holds a request, and the command asks for another. The
request on disk may carry edits, so it is never re-derived silently: edit it,
use another `--workspace`, or `--clean` to derive it again.

### `argument --goal: expected one argument`

A goal that starts with `-` (`-mtp tp1`) reads as an option. Write
`--goal=-mtp tp1`, or `no-mtp tp1`.

## The planned run

### `MISSING <unit>:<gate>@<member>` for a check that passed before

A verdict holds for the state of the target its check measured: a fingerprint
of the commit it ran on, less what no gate runs or reads. A commit under
`tests/`, `docs/`, `.github/` or `examples/`, or to a `*.md` file, leaves every
verdict standing, unless it changes a file a gate reads there (the quickstart,
the default boot image, a file the spec's `accuracy`, `runtime`, `variants` or
`environment` block names); so does an edit to another model's entry in an
anchor table. A native rebuild makes every verdict stale, since the libraries'
sizes and modification times are part of the fingerprint. Any other change
makes a verdict stale when it touches what its check read: a non-Python file
of the tree, or a Python module one of its processes imported; a verdict that
recorded no read list counts as missing after any such change
([design.md §5.5](docs/design.md#55-fingerprints-and-staleness)). An
uncommitted change in the checkout moves no verdict. The checks a finished
task claimed run again once no task is running.

`SHARED <check>: holds by the pass of <other>` in the same standing is not
missing: the two run one command on one engine against one criterion, so one
pass on this tree serves both. A failure is never shared.

### `[VOID] <gate>@<member>`, or `VOID <check>` in the standing

The check ran on the snapshot of one commit (`<workspace>/snapshots/<commit>`),
but a command printed that it imported the package from somewhere else
(`imported` in `contract_status.jsonl`, one entry per command). A path is a
checkout no commit pins, usually the integration checkout; two commits are two
snapshots. The check was measured once more before it was recorded as not
passing. Find what put the other tree ahead of the snapshot on `sys.path`: a
`PYTHONPATH` in `environment.env`, a `contrast_command` that changes directory
or sets its own `PYTHONPATH`, or a package installed into the interpreter
rather than linked to a checkout. `git worktree list` in the checkout shows the
snapshots that exist.

### `the checkout holds ... uncommitted; this check measured <commit> without them`

A warning: someone edited the integration checkout instead of a task worktree.
Commit the change through `bin/commit.sh` to have it measured, or remove it.

### `failed on attempt 3 of 3 and is not retried again`

The planner wrote a failed task back as `pending` once more than it may; the
revision is refused, and the planner is told to find the cause (an
`investigate` task), reshape the work, or drop the task. A failure of the
run's infrastructure (the turn raised, `INTEGRATION_FAILED`, a tune baseline
that could not be measured) costs no attempt: the task starts again once by
itself.

*Caught by:* `apply_revision`.

### `INTEGRATION_FAILED`: `conflicts with the integration head ... in: <files>`

The task's branch did not rebase onto the checkout's head, and its worker,
sent back twice to resolve, could not. The task starts once more by itself;
after that, merging its branch is an `integrate` task. Two tasks that edit one
file conflict this way: name files in `touches`, so the scheduler keeps them
apart.

*Caught by:* `rebase_block`.

### `[FAIL] ops:promoted: <path> is a local variant of <category>/<name>`

Delivery found a local variant still in the target's `ops/`, or a target file
still importing one, so no promote ran after the last variant landed: the last
one failed (often two variants that changed `__call__` differently, or a cell
that passes alone and fails merged), or a variant landed after it. Retry or
reshape the promote, or name the entry in `target.keep_local`. A task that
imports a variant after the promote removed it fails with an `ImportError` on
the variant's module: point that import at `catalog/<category>/<name>`.

*Caught by:* `check_promoted`.

### `every required check holds, but no regression check covers <tasks>`

Those tasks landed changes to shared code that no passed regression check
covers, usually because the check that covered them failed: a failed check
covers nothing, and no new one is added for what already landed. A fix that
changes shared code is followed by a new check, whose batch takes every
uncovered change; else the planner retries the failed one (`status: pending`)
once what it ran on has changed.

*Caught by:* the orchestrator, before it delivers.

### `<test> (passed at the base, fails now)`, `(new, fails)` or `(no result at the head)` from a regression check

`task <id> (regression check) ran the package tests the changes to shared code
it covers (...) can affect; what failed: ...` is the planner's event for a
failed regression check, which ran the affected modeling_v2 package tests in a
snapshot of the head against the commit the last check cleared:

- `(passed at the base, fails now)`: a change in the batch broke it;
- `(new, fails)`: a test the base did not have fails at the head;
- `(no result at the head)`: the job produced no result for it (crashed, timed
  out, not collected);
- `(fails at the head; no result at the base)`: the base could not be measured;
- `the test job recorded nothing (exit <n>): ...`: the job failed before pytest
  recorded a session.

An `investigate` task finds the cause; a worker reruns the same tests with
`trtllm-modeling-v2 tests affected` and `tests compare`. Never loosen the test
to make the check pass.

*Caught by:* `run_regression_check`.

### A regression check whose dependencies are done does not start

It waits on purpose: one regression check runs at a time, and while a task
that changes shared code runs, a ready one waits up to 30 minutes after its
batch's first landing so that change joins it. `task <id> (regression check)
closed: no change to shared code was left for it to check` is not a failure:
the checks before it covered every change.

### `the engine's greedy tokens are not the HF reference's (<file>): <prompt> step <n>: engine token <a>, reference <b> (...)`

The boot sent the logit reference's text prompt as token ids, and the engine's
greedy token at step `<n>` is not the reference's and not a near-tie. Read it
as the first token a defect changed: a wrong layer, a misread weight, a wrong
position, a speculative verify that accepts what the target would not have
generated. Run `trtllm-modeling-v2 gate logits` at the same member for the
per-position numbers, and the same boot under `--mode off` to see whether the
stock path agrees. `generated <k> of <n> tokens` is an engine that stopped
before its budget: a dropped request or a length limit.

*Caught by:* `gates.judge_greedy`.

### `the boot's greedy reference: ...`, or `no text prompt of the greedy reference ... fits this unit's engine`

The run names a logit reference, so every boot compares its greedy tokens, and
this boot could not: the file is missing or not a reference, or every text
prompt in it is longer than the unit's `max_num_tokens` without chunked
prefill, or than `max_seq_len`. Make a reference with shorter prompts
([user-guide.md §2](docs/user-guide.md#2-prerequisites)), or set
`accuracy.boot.greedy.enabled: false`.

*Caught by:* `gates.greedy_workload`.

### A unit's benchmark is owed at fewer members than it has

With the logit gate owed, `accuracy[<benchmark>]` runs only at the narrowest
member, the widest, and every member the logit gate cannot stand for; the
others owe `logits`. `contract.yaml`'s `notes` say where each unit's benchmark
runs. `accuracy.benchmark_members: every` asks for it at every member.

---

## Performance measurement (the tune stage)

### `TUNE_FAILED (tune-failed:smoke)`: `the perf smoke failed; the baseline is not started`

Before its baseline, the tune task serves each world size once (concurrency 1,
4 prompts of 32 output tokens, a 45-minute wall, `perf.campaign.smoke_wall`).
The line names the world size and its perf config, then the job's return code
and log (`rc=<rc>, see <log>`) or ``served `modeling_v2: off`: the perf config
does not select the target``. Fix the launch or the config and retry the task;
`perf.campaign.smoke: false` skips the smoke.

*Caught by:* `smoke`.

### `[tune] <label> on disk ...; set aside as <label>.stale-<time> and measured again`

Not a lost result. A measurement on disk is reused only when its `meta.json`
says it measured the same workload, the same merged LLM-API config and, on the
campaign's own tree, the same commit (`stale_measurement` in
`perf_ledger.jsonl`).

### nsys: `Profiling started at iteration N`, then the server sits at 0% GPU and no report is written

`torch.cuda.cudart().cudaProfilerStop()` (what
`tensorrt_llm/_torch/pyexecutor/profiling.py` calls) holds the GIL while CUPTI
waits for GPU work other Python threads must complete, and the driver's
`cuProfilerStop` through ctypes did not return either. Use
`bench.py --nsys <prefix>` in its default session mode: `nsys launch` +
`nsys start/stop` timed from outside (`--nsys-delay`, `--nsys-duration`);
nothing in the server calls the profiler API.

### A measurement shares its GPU with a server nobody is running

A cancelled job's `nsys profile` (or any server started in its own session) is
re-parented to init and outlives the job's leader. The scheduler finds such
processes by the markers in their environment and stops them; a process started
outside the scheduler carries none. Find processes whose parent is 1
(`ps -eo pid,ppid,comm | awk '$2==1'`) and kill them by pid, never by a pattern
that also matches your own shell. `bench.py` refuses a device with more than
2 GiB in use.

*Caught by:* `processes_with_marker`.

### Every probe on one node looks 3-4% worse at concurrency 1

The node, not the knob: identical configs differed by up to 4% between GB300
nodes while same-node repeats agreed within 0.1%. Judge against a same-host
control, as the campaign does.

## Python environment

### The import resolves to a different tree than the one you are measuring

`/root/.local` and `/root/.cache` may be symlinked into a shared home, so
another checkout's develop-install is on the path of every container run.
Export `PYTHONNOUSERSITE=1` at runtime, **but not during the native build**: the
CUTLASS kernel CMake runs `setup_library.py develop --user` and then imports
`cutlass_library`. A worktree has the same symptom with another cause; see
`reason=mixed_checkout`.

### The gate measures the wrong build entirely

Point `environment.python` at the editable install you intend to measure:
resolving the interpreter once, in the spec, keeps a run from gating a
different build.

### `incremental_build: <python> is not Python >= 3.7; set MV2FLOW_PYTHON to the run's interpreter`

No build started (exit 64): the rebuild script runs its helper under
`$MV2FLOW_PYTHON`, else the `python3` on `PATH`, and a login node's `python3`
can be older. Point `MV2FLOW_PYTHON` at the run's interpreter
(`environment.python`).

*Caught by:* `check_host_tools`, as a preflight note on a `local` run.

---

## The agent plane

### `You cannot dynamically configure MCP servers when an enterprise MCP config is present`

The presence of `/etc/claude-code/managed-mcp.json` makes the SDK refuse every
dynamically registered server; the first symptom is otherwise an opaque
`ProcessError: exit code 1` on the first turn. The workflow registers no MCP
server of its own (roles report through decision files), so the error comes
from an `extra_mcp_servers` entry in the `agents` block, which such a host
refuses.

### `API Error: EROFS: read-only file system, mkdir '.../credentials.lock'`

Reported as `stop_reason='stop_sequence'` with **zero tokens**, so it reads as
a model refusal; no agent started. The spawned CLI's first act is to take a
lock beside its credential store, and a sandboxed `$HOME` is read-only. Use
`--agent-transport external`
([README](README.md#running-where-the-orchestrator-cannot-authenticate-an-agent)).

### `Not logged in · Please run /login` on a turn that used to work

The second half of the entry above: a credential store copied to a writable
home holds a copy of a refresh token the parent session also holds, and a
refresh rotates it, so one copy loses (measured: authenticated at 12:12,
failed at 12:25, byte-identical credentials). Use `--agent-transport
external`: the turn becomes a file, an already-authenticated session runs it,
and everything the orchestrator decides stays in Python. A driver of your own
must deliver **the role's system prompt** with the turn (`<id>.system.md`).

*Caught by:* `probe_agents`, on the in-process transport: the probe's turn
fails with this text, and the run stops before its first planner turn.

### `... is older than the minimum version required by your organization`

Every in-process turn is refused at start-up by the organization's policy: the
Claude Agent SDK launches the `claude` it bundles, which can be older than the
CLI you use by hand. The exception may only say `Command failed with exit code
1`; the text is in the CLI's stderr. Set `AGENT_FLOW_CLAUDE_CLI` to the
`claude` that works in your shell (`command -v claude`) and start again.

*Caught by:* `probe_agents`, which reports the CLI's stderr and the binary it
launched.

### `connect EPERM ... srt-mux-*.sock`, in a turn that ended normally

The turn's Bash tool could not start its own sandbox: the orchestrator was
launched from inside another agent's session (`SANDBOX_RUNTIME` or
`CLAUDECODE` is set), and a sandbox cannot be started from inside one. The
model reported that Bash failed and ended its turn as if it had worked. Launch
the orchestrator from a plain terminal or a tmux session, or use
`--agent-transport external` and let your session drive the turns.

*Caught by:* `read_probe_file`: the probe's file holds the nonce's SHA-256,
which only a shell that ran writes.

### `agent probe: N of M agent(s) cannot run a turn with Bash here`

Preflight's agent probe failed, and the run stopped before its first planner
turn. Under it, per agent: what failed, what the turn said, the CLI's stderr,
and its files (`gates/agent-probe/<n>/<id>.*`), then what to do: the entries
above, a login, or `--agent-transport external`. `the turn's shell ran, but
git -C ... rev-parse HEAD printed ...` means the turn cannot read the checkout
(another mount namespace, or git's `dubious ownership`). Set
`MV2FLOW_SKIP_AGENT_PROBE=1` only when the probe itself is what is broken.

*Caught by:* `probe_agents`.

### `<id>: no answer within Ns (the <role> turn timeout); the turn was stopped`

An in-process turn ran past its role's bound (`--agent-turn-timeout`). It was
cancelled, which stops its CLI, and raised `TurnTimeout`: a task's turn is
retried once, then the task fails to the planner. What it did is in the
activity log the message names.

*Caught by:* `InProcessTurnExecutor`.

---

## Filesystem and git

### Everything stops at once with unrelated-looking errors

Check `TMPDIR`. Engine builds, NVRTC scratch and harness output default there,
and on a shared host another tenant can fill `/tmp` mid-run.

*Caught by:* `check_tmpdir`, which rejects a `TMPDIR` under `/tmp`, reading the
value **declared in `task.yaml`**, not the orchestrator's own environment.

### `ipc path "..." is longer than 107 characters`

Followed by `zmq.error.ZMQError: File name too long`. 107 is
`sizeof(sockaddr_un.sun_path)`, a kernel cap on unix socket paths. The LLM
API's RPC binds an IPC socket under `TMPDIR` named `rpc_test_<uuid4>`, 45
characters plus a separator, so **`TMPDIR` must be at most 61**. It is the
length of the name, not the filesystem: unix sockets bind fine on Lustre, and a
`TMPDIR: /tmp` workaround works because `/tmp` is short. Point `TMPDIR` at a
shorter path on the same storage.

*Caught by:* `check_tmpdir`, which derives the bound from
`UNIX_SOCKET_PATH_MAX` and `RPC_SOCKET_BASENAME_LEN` and shows the arithmetic.

### `Unable to create '.git/index.lock': File exists`

Two agents committing in the same instant, or a stale lock from a crashed
process. Before removing anything, confirm no git process runs (`pgrep -a
git`); a zero-byte lock with no owning process is stale. Agents commit through
`bin/commit.sh`, which takes an `flock`: its exit code **75** means it waited
for the lock and did not get it, a queue, not a broken repo.

### `commit.sh: REFUSED: ...` (exit 2) and `REFUSING TO LEAVE THIS COMMIT AS IS` (exit 1)

Exit 2 is a refusal before anything was staged; fix the call, not the repo:

- `is a symlink git does not track`: a worktree's link to the main checkout's
  native build (`tensorrt_llm/bindings` and friends). Name real files only.
- `sandbox placeholder`: `.mcp.json`, or anything under `.claude/` or
  `.cc-writes/`, which an operator's sandbox leaves wherever its shell goes.
- `outside the paths this run may commit`: `commit_policy.allowed_paths`
  (default `tensorrt_llm/ tests/ cpp/ examples/`).
- `Co-authored-by` / `Signed-off-by` in the message: the script signs off as
  the repository's user itself, and the repository takes no co-authors.
- `no commit identity`: set `user.name`/`user.email` in the repository, or
  export both `MV2FLOW_GIT_NAME` and `MV2FLOW_GIT_EMAIL`.

Exit 1 with `REFUSING` means the commit was made and undone: the file list
differed from the request, a symlink was added, a sign-off did not carry the
author's email, or a `Co-authored-by` appeared (usually a hook or an
environment variable, never the agent). Delivery runs the same checks over
every commit since the merge base (`git:commit-hygiene`); a commit missing a
sign-off has to be re-signed as its own author.

### `could not lock config file .git/config: File exists` and the lock will not delete

Seen on Lustre with `Device or resource busy`. Pass the identity on the
command instead of writing it to the config:

```bash
git -c user.name="..." -c user.email="..." commit -s ...
```

---

## Measurement traps

These produce **numbers**, not errors, which makes them the most expensive
category: a wrong number is believed.

### The accuracy gate reports a large regression that is not there

Measured once: **63.50 against a 95.53 anchor, a 27-point phantom**: the anchor
had been scored with a chat template and the gate had not. A benchmark's
protocol is part of its identity. Declare it:
`accuracy.benchmark_args: [--apply_chat_template]`.

### A feature's gate re-measures the base path

The `mtp` unit's accuracy gate must carry a `speculative_config`; without one
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
criteria silently gets the built-in implementation, and the number reads as
modeling_v2's. Anything you attribute to modeling_v2 runs under `require`,
which raises instead, exported **before the ranks start**: workers receive the
environment as it stood at MPI init.

### `the engine did not run the declared configuration -- <field>: declared X, engine Y (read from ...)`

The gate built its engine as the unit declares, and the engine ran something
else. Each field names where its value was read: `worker` (rank 0's executor
after it loaded the model), `llm.args` (the validated LLM-API arguments), or
`engine log` (a rank's own warning). Measured once: block reuse off in every
quickstart gate of a hybrid model for about 30 hours while each gate passed.
The fix is in the declaration:

- `block_reuse: declared on, engine off`: for a hybrid model, set
  `kv_cache_config.mamba_state_config` in the unit's options file; otherwise
  find the engine's warning in the log, and turn `kv_cache_reuse` off for the
  unit if the engine is right.
- `cuda_graph: declared 19 batches ..., engine 34 batches ...`: two launch
  paths resolve different lists; `--dry-run`'s `check_capture_lists` names the
  unit.
- `kv_cache_dtype: ... engine asked for nvfp4 and resolved FP8`: the worker
  fell back (NVFP4 KV on SM107, for one); the device cannot serve the declared
  dtype.
- `modeling_v2: declared off, engine require`: the tree's quickstart passes its
  own `modeling_v2`, and the harness adds the gate's only when it is absent
  (`engine_kwargs` in `checks/harness/engine_driver.py`).

A field the record does not carry shows as `NOT ATTESTED <field>`, never as
agreement. Where rank 0's worker runs in another process, block reuse is read
from what the engine reused (`read from reuse probe`, `read from cached
tokens`).

*Caught by:* `attest_engine` (`checks/gates.py`), over the harness's
`MV2FLOW_ENGINE` and `MV2FLOW_REUSE` records.

### `the unit asks for KV-cache block reuse and the engine reused nothing: ...`

The engine resumed no request from its KV cache although it had a prefix to
resume from: the reuse probe's second prompt was computed from token 0, or more
requests than the engine runs at once shared a long enough prefix and none
resumed. The configuration can say reuse is on and still reuse nothing.

- A hybrid model resumes only where it holds a recurrent-state snapshot: check
  `kv_cache_config.mamba_state_config`, and `mamba_snapshot` in the same
  `[engine ...]` segment.
- A speculator whose `context_recompute_tail` is `-1` recomputes every resumed
  prompt; turn `kv_cache_reuse` off for that unit if this is intended.
- `reuse probe not read (...)` means the probe did not run (`skipped`: it does
  not fit `max_seq_len`; or the error it raised), and nothing is judged from it.

*Caught by:* `reuse_shortfall` (`checks/gates.py`), in the boot gate, both
acceptance legs, both envelope legs and the accuracy gate.

### `[WARNING: LOW MARGIN: 0.52 GiB free on device 0 ... once the engine was up]`

The gate passed with little device memory left once its pools were sized and
its CUDA graphs captured; the next change that adds memory to that
configuration fails at model creation in a unit nobody touched. Measured once:
MTP depth 7 at TP1 ran with 0.515 GiB free, and when the vision tower landed
(+3.26 GiB per rank) it ran out of memory, 7.5 hours later. Recheck the
configuration with the least free memory first after an integration, and find
what holds memory twice. The threshold is the larger of
`runtime.min_free_memory_gib` (default 2) and
`runtime.min_free_memory_fraction` (default 0.02) of the device.

*Caught by:* `MemoryMargin` (`checks/gates.py`). It warns; the verdict stands.

### Two legs of one configuration differ in a token comparison, and neither leg is wrong

Each leg is a separate process, and a `@triton.autotune` kernel picks its
config by timing, per process; two configs reduce in a different order, so one
configuration can produce two outputs. Measured once: two output modes of one
configuration, matching the `num_warps` one FLA chunk kernel's autotune picked
in each process, with one prompt first differing at token 0. Every leg of a
gate run shares the AutoTuner's tactics and Triton's autotuning, emptied at the
start of the run. If legs still differ, run a leg with
`TRITON_PRINT_AUTOTUNING=1` twice and compare the configs it prints. Do not set
`TRITON_CACHE_AUTOTUNING` and `TRITON_CACHE_DIR` in `environment.env`: that
changes every gate, and the gate's own pins override them anyway.

*Caught by:* `pin_tuning`.

### `INSENSITIVE, not a pass: ... with the received state destroyed (...) continuation #N (W words) was still token-identical`

Leg B ran its longest prompt again with `MV2FLOW_WITNESS_ABLATE=state`, and the
output did not change although the state the generation engine received was
destroyed: a workload that does not read that state passes a hop that drops it.
Measured once: with a hybrid model's recurrent state zeroed after the hop,
prompts of `one two ... ten` stayed identical, while prose prompts diverged at
tokens 1 and 11.

- Replace a prompt that repeats itself or is a famous text with prose the
  model has to read. `prompt_tokens` generates prose (`checks/witness_text.py`),
  but past the passage's length (about 5400 words) it repeats the passage; for
  a longer prompt, declare prose of your own in `prompts_file`.
- Leg B printed `Witness ablated:` but destroyed nothing that matters: check
  that its hook zeroes the slot the request was given
  (`_get_mamba_slot_for_request`) on the generation engine, on every rank,
  after the transfer completed and before the first decode step.

*Caught by:* the self-test in `run_equivalence_witness` (`checks/witness.py`).

### ``the self-test could not run: the self-test leg (MV2FLOW_WITNESS_ABLATE=state, ...) printed no `Witness ablated:` line``

- The block's `contrast_command` does not implement `MV2FLOW_WITNESS_ABLATE`:
  it is the run's code, and must honour the contract in
  [design.md §5.2](docs/design.md#52-gates-and-witnesses), with the hook in a
  `sitecustomize` on every engine process's `PYTHONPATH`.
- The launcher installed its hook, but the tree under test hands a received
  request to generation somewhere other than where the hook acts
  (`PyExecutor._prepare_disagg_gen_transmission_complete` in the current
  engine). Find where, and move the hook there.

### `the disaggregated leg did not complete`

Leg B, the block's `contrast_command`, failed or ran past the witness's
`timeout`; the verdict's tail is the end of its output. A disaggregated launch
hangs more often than it raises. Check, in order: the `cache_transceiver_config`
it was handed as `MV2FLOW_WITNESS_TRANSCEIVER` (both engines need the same one;
`DEFAULT` resolves to NIXL unless an environment variable selects another
backend, and the Python transceiver runtime drives only NIXL), the interpreter
`MV2FLOW_WITNESS_ENGINE_LAUNCH` names, and the devices each engine got.

*Caught by:* `run_equivalence_witness`.

### `leg B did not report pinning each hop to its attention-DP rank pair (MV2FLOW_WITNESS_RANK_PAIRS=...)`

At a `dep` member, leg B must pin prompt `i`'s context-only and
generation-only requests to the `i`-th pair
(`SchedulingParams(attention_dp_rank=..., attention_dp_relax=False)`) and print
`[i] Witness ranks: ctx=<c> gen=<g>`. Unpinned, a leg that sends one request at
a time crosses every hop from rank 0 to rank 0, and a transfer defect on any
other rank is never compared. `reported ctx=0 gen=0 for 0:1` means the
launcher printed the line without pinning.

### `llm.get_stats()` returns nothing

The feature witnesses for overlap scheduler, chunked prefill and block reuse
are **unverified for this reason**: a self-diagnosing witness reported
`[stats] 0 iterations`. Earlier attempts also failed for unrelated reasons
(concurrent `MPI_Init`, `TLLM_DISABLE_MPI=1` routing to a missing Ray, wrong
stats field names returning zeros).

*Open.* If you need these features proven, this is the thing to fix first.

### The survey says good catalog entries are "Rotted — do not spend these"

```
**Rotted** — the entry names an `impl` that does not resolve in this build.
- `comm/allgather` → `torch.ops.trtllm.allgather`
- `quantization/fp4_quantize` → `torch.ops.trtllm.fp4_quantize`
- `attention/fused_qk_norm_rope` → ...
```

Check one by hand before believing it: **`torch.ops.<ns>` is a lazy proxy**, and
`dir()` lists only the names something has already resolved. Measured on a
GB300 build:

```python
len([n for n in dir(torch.ops.trtllm) if not n.startswith("_")])   #  91
len({s.name for s in torch._C._jit_get_all_schemas()
     if s.name.startswith("trtllm::")})                            # 273
```

*Caught by:* `survey.py`, which enumerates from the schema registry and records
`op_enumeration: schema_registry | dir_fallback`.

### A feature unit's gate fails to start with `no such option`

The evaluation protocol is options of the benchmark subcommand, and the
subcommands do not share an option set: `gsm8k` takes `--apply_chat_template`
and `--fewshot_as_multiturn`, `mmmu` takes neither. A run-wide
`accuracy.benchmark_args` makes a multimodal unit's gate fail to start with a
click usage error. `benchmark`, `reference_yaml`, `llm_api_options`,
`discriminators` and `benchmark_args` all take the per-unit mapping form; use
it.

### Two `task.yaml` copies disagree

The spec is copied into the workspace, and `_refresh_workspace_spec` copies it
again on every resume. To change a spec mid-run, edit the source and reload or
resume; do not edit the workspace copy.

### `it ran in another environment`, `opaque: ...` or `environment not checked` from `tests affected` / `tests compare`

The test cache reuses a result only where nothing the test ran on changed, and
says why it did not, per test:

- `it ran in another environment`: the interpreter, installed packages, GPU
  models or count, or a `TLLM_`/`TORCH_`/`NCCL_`/`CUDA_`-style setting differs
  from the recording's. Run the tests in the same image on as many devices each
  time.
- ``opaque: imported `tensorrt_llm` from outside the tree``: the tests imported
  an installed copy instead of the checkout. Run from the checkout's root, as
  the printed command does.
- `opaque: starts ...` or `hands work to other processes`: from that test on,
  its process runs code the recorder cannot see, so its results are reused only
  at the commit they ran on.
- `its commit is not in this repository`: `--repo` is a clone without the
  recorded commit; name the checkout the run integrates into.
- `environment not checked`: nothing was recorded at the head; run the printed
  driver command without a selection, which records only the environment.

*Caught by:* `head_environment` (the environment check); the rest is the cache
declining to guess.

---

## When none of this fits

Say so rather than working around it. The agent prompts carry this
instruction: *"If this prefix does not fit what you need to run, that is a
**symptom to report in your final message** — not a problem to solve"*, and
the same applies to a human operator.
