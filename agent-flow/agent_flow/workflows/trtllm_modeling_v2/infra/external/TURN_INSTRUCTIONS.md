# How to execute one trtllm-modeling-v2 role turn

You have been handed a **request id**. Everything else is in this file.

The `trtllm-modeling-v2` orchestrator is running as a plain Python process that cannot
authenticate a child agent on this host, so it has written your turn to disk and is blocked
waiting for your answer. You are the transport and nothing else: you do the role's work, you do
not second-guess the orchestrator.

**Working root: `$root`.** Every relative path anywhere below, including inside the prompt files,
is relative to that directory. Use absolute paths or `git -C`; **do not `cd` into subdirectories
of the repository**: a sandboxed shell leaves placeholder files (`.mcp.json`, `.claude/`) in every
directory it visits, and they dirty the tree.

## 1 · Find your turn

```bash
$driver prompt <your-request-id>
```

That prints the resolved absolute path of every file this turn involves, your spool job prefix,
and, only when the run routes the role to a particular agent, that agent. Otherwise run the turn
on your own model and reasoning effort. Use those paths; do not guess them.

## 2 · Load your role

Read `<id>.system.md` **in full, first**. Every instruction in it is binding on you for this turn,
exactly as if it had been installed as your system prompt. Then read `<id>.prompt.md`, which is
the turn itself.

The run's skills are under `$skills`. They are **not** auto-loaded for you. Read the ones your role
prompt names as ordinary files, including their `references/` subdirectories.

If the request lists `prior_turns`, you have held this role before in this run. This transport
starts a fresh session per turn, so read those turns' prompt and result files and treat them as
your own prior reasoning.

## 3 · Do the work

### Running GPU work

Compute sessions are already running, each watched by a spool scheduler. You do **not** submit
Slurm jobs; `srun` and `sbatch` do not work from here. Work is a file:

$sessions

```bash
export MV2FLOW_JOB_PREFIX=<your prefix>          # printed by `prompt`; names must start with it
SPOOL=<spool> $spoolctl submit --gpus <N|all> --wall <limit> [--tree <checkout>] \
    <prefix>-<what> '<command>'                    # runs it, waits, prints the log
SPOOL=<spool> $spoolctl launch ...         # same options; queues it and returns
SPOOL=<spool> $spoolctl status          # what runs, what waits and why
SPOOL=<spool> $spoolctl cancel <name>   # the ONLY way to stop a job
```

- **`--gpus`.** The scheduler allocates devices and exports `CUDA_VISIBLE_DEVICES` to exactly
  those; never set it yourself. `--gpus all` holds the whole node exclusively: use it for anything
  that loads the full model. A waiting whole-node job is never starved by single-device jobs.
- **`--wall`.** Counted from the job's start; queued time is free. A job past its limit, or one
  whose log goes silent after a traceback, is stopped and its receipt says why.
- **`--tree <checkout>`.** Runs the job with that checkout as its working directory **and** first
  `PYTHONPATH` entry, so every MPI rank imports it. A result whose processes imported
  `tensorrt_llm` from different checkouts is refused (`reason=mixed_checkout`). Always use it for a
  worktree.
- **Names** start with your request's prefix (`r014-...` for request `014-kernel_writer`). The submit
  script refuses any other name, so you can never overwrite another turn's results.
- **Which spool.** A spool that will not start the job soon (its session is in hand-over, ends
  before `--wall`, or has no live watcher) gets a one-line `warning:` naming the spools of the same
  kind (`work-*`, `gates-*`, ...) that can take it now; `--pick` sends the job to whichever of them,
  the given spool included, can start it soonest and prints which, so `wait`, `status` and
  `cancel` go there.
- **Native rebuilds** go through
  `$spoolctl submit --gpus all --wall 3h build-<what> 'bash $scripts/incremental_build.sh --repo <checkout>'`:
  it announces the rebuild on every session and gives every library a fresh inode first.
- **Do not hand back while a job you started is still running.** Wait for it, or cancel it and
  say so.

### Hard rules inside a session

- **The safe-kill rule.** Never kill, time out or crash a process attached to the allocation's MPI
  (one that carries `PMI*`/`PMIX*`/`OMPI*`/`SLURM*`): Slurm's MPI error handler then kills the whole
  allocation. The scheduler removes that context from every job and stops jobs itself;
  `spoolctl.py cancel` is the only way to ask it to.
- **Never `pkill`/`killall`/`kill` by pattern.** The scheduler is the allocation's main process;
  killing it destroys the session and every queued job.
- Export `PYTHONNOUSERSITE=1` in anything you run.
- `TMPDIR` must stay short (an RPC socket is bound under it): `$tmpdir`.
- A multi-rank engine launch from inside a spool job uses the launch line your role prompt gives;
  copy it whole.
- `UCX ERROR uct_iface_open` lines are noise.

### The source embargo

$embargo

### Commits

$commits

## 4 · Hand the turn back

1. write your final message (the prose you would have ended the turn with) to `<id>.result.md`;
2. **then** write the single character `1` to `<id>.done`.

Never write `.done` first. If you are blocked, still write both files: put the blocker in the
result file and, if your role prompt defines a decision value for that case, use it. A turn that
ends without `.done` hangs the run until its timeout.

## 5 · Report back

Three lines to whoever dispatched you: the decision you wrote, what you produced, and anything the
orchestrator should know that did not fit in the decision file.
$local
