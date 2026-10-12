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
| `claim_devices` | enough idle devices of the declared arch, by free memory and a lone-process probe |
| `check_multirank_substrate` | NCCL collective + MPI spawn answer under a deadline, for a multi-rank run |
| `substrate_gate` | **this host generates a token** on the stock path under this run's exact runtime block |
| `agent_plane_gate` | one agent turn can complete at all |

`substrate_gate` is the one that earns its minutes. It *generates* rather than
constructs, because the NVRTC failure below appeared on the first generated
token — after the engine had loaded 22 GB of weights, allocated the KV cache
and finished warm-up. A gate that stopped at "the engine constructed" passes
it straight through.

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

*Caught by:* `_mpirun_launch`, which adds `--oversubscribe` on the `mpirun`
branch only. Under `slurm`, `srun --ntasks` genuinely sizes the world, so the
same message there is a mis-sized allocation reporting itself and the flag
would hide it.

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
The `slurm` branch keeps today's behaviour, where those variables are correct.

### `UCX ERROR uct_iface_open(tcp/rdma_vf_rail0) failed`

Noise, on every rank, and a documented red herring. UCX fails to open one IPv6
RDMA virtual-function interface, falls back, and the run completes with exit
code 0 and correct output. `UCX_TLS=sm,self` silences it and fixes nothing.

*Not caught, and deliberately not.* If the run is also hanging, the cause is
the entry above, not this.

### Two jobs die together, both at `MPI_Init`

Concurrent `MPI_Init` in the same allocation killed two compute sessions in
this run. If you are dropping several scripts into a spool at once, either
serialize the ones that initialize MPI or give them distinct allocations.

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
MCP broker is the only path. **Do not conclude the cluster is down.** Set
`environment.kind: slurm_broker`.

One consequence is worth knowing before you choose it: the orchestrator is not
an MCP client, so gate *execution* moves to an agent while gate *adjudication*
stays in Python — the orchestrator reads the log off the shared filesystem and
applies its own thresholds, so no verdict rests on an agent's report of what a
number was.

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
import **per job**, is why `session_spool` exists and why it is the kind to
prefer wherever available:

- one allocation, acquired once, running a watcher as its own main process;
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

## Python environment

### The import resolves to a different tree than the one you are measuring

`/root/.local` and `/root/.cache` are symlinked into a shared, relocated home
on this cluster, so another checkout's develop-install is on the path of every
container run. Export `PYTHONNOUSERSITE=1` at runtime.

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

The workflow **detects this and switches to file-based decisions**
automatically (`ENTERPRISE_MCP_CONFIG` in `workflow.py`); each role writes
`decisions/<role>-N.yaml` instead of calling a tool. Nothing to configure.

### `API Error: EROFS: read-only file system, mkdir '.../credentials.lock'`

Reported as `stop_reason='stop_sequence'` with **zero tokens**, so it reads as
a model refusal. It is not: no agent ever started. The spawned CLI's first act
is to take a lock beside its credential store, and a sandboxed `$HOME` is
read-only.

*Caught by:* `agent_home.py`, which stages a writable home under the workspace
and exports `HOME` + `XDG_CONFIG_HOME` + `CLAUDE_CONFIG_DIR` before any agent
is constructed. Four details each cost a cycle to find, so do not "simplify"
them away:

- `HOME` alone is not enough — the credential store resolves through **XDG**;
- the config directory must **not** be named `.claude`, because a sandbox that
  denies `**/.claude` writes cripples the child with a *warning*, not an error;
- the path must be **absolute** — a relative `HOME` resolves against the
  *child's* cwd and silently builds `<workspace>/<workspace>/agent-home/`;
- `$HOME/.gitconfig` has to be written, or every commit dies with *"Please
  tell me who you are"* — a new infrastructure failure traded for the old one.

- a **symlinked** `$HOME` does not work — the lock follows the resolved path.

### `Not logged in · Please run /login` on a turn that used to work

This is the *second* half of the problem above, and staging a home does not
fix it. The staged credential store holds a **copy** of a refresh token the
parent session also holds, and a refresh rotates it — so one of the two copies
loses. Measured on this cluster: authenticated cleanly at 12:12, failed at
12:25, byte-identical credentials. A previous run measured "about forty
minutes". Ten concurrent smiths make it worse, not better.

Copying harder does not fix a single-use token.

*The answer is `--agent-transport external`.* The turn becomes a file: the
orchestrator writes `<workspace>/dispatch/<id>.{system,prompt}.md` and blocks
until a driver writes `<id>.result.md` and then `<id>.done`. Any
already-authenticated session can be the driver. Everything the orchestrator
*decides* stays in Python — the stage machine, the queue walk, the dependency
batching, decision-file validation and all three gates.

One thing to check if you write your own driver: **the role's system prompt
travels with the turn** (`<id>.system.md`). A driver that delivers only the
user prompt is running a different agent — no role, no invariants, no embargo
— and its answer still looks plausible.

*Caught by:* `agent_plane_gate`, which spends one trivial turn in preflight so
this costs seconds rather than a 24-minute planning turn.

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

Under `TRTLLM_MODELING_V2=auto`, a configuration that misses a target's routing
criteria silently gets the built-in implementation — and the number reads as
modeling_v2's. Anything you will attribute to modeling_v2 runs under
`require`, which raises instead of falling back.

It must be exported **before the ranks start**, not merely before `LLM(...)`:
workers receive the environment as it stood at MPI init, so a driver resolving
a modeling_v2 target while its workers resolve the built-in is exactly the
silent split the switch exists to prevent.

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

*Caught by:* `survey.py`, which now enumerates from the schema registry and
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

---

## When none of this fits

Say so rather than working around it. The agent prompts carry this instruction
verbatim — *"If this prefix does not fit what you need to run, that is a
**symptom to report in your final message** — not a problem to solve"* — and
the same applies to a human operator. A run that spends its budget
rediscovering how to submit a job is the failure mode this whole workflow was
shaped to avoid.
