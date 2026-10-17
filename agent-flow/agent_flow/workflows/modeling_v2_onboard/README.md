# modeling-v2-onboard

Bring up a new model, or a new target of an existing model, in ModelingV2
(`tensorrt_llm/_torch/_experimental/modeling_v2/`). Two roles:

- **planner** — reads the task, the checkpoint and the catalog, and writes
  `plan.md`: tagged tasks plus an `## Execution Graph` of their dependencies.
  It never writes code. It runs alongside the workers: whenever one finishes,
  it is woken with the result and the free worker slots, and extends or
  revises the plan while the others keep going. Its aim is to keep
  `--max-parallel` workers busy: independent work side by side, feature work
  alongside the accuracy gate, and `perf` tasks in the slots left over.
  Checks never block the next step: a join waits only for the code it merges,
  and a failed check gets a fix task and a new join.
- **worker** — runs one task in its own git worktree and ends by writing
  `task_results/task_result_<id>.md`. Workers run in parallel.

```bash
modeling-v2-onboard --task task.yaml --workspace /scratch/me/qwen3-onboard
modeling-v2-onboard ... --backend codex          # every agent on Codex
modeling-v2-onboard ... --max-parallel 4         # at most 4 workers at once
modeling-v2-onboard ... --no-primer              # op workers start afresh
modeling-v2-onboard --workspace /scratch/me/qwen3-onboard \
    --feedback "Drop the regression tasks; they are not needed."   # steer a running run
```

The run's agents do not read or write Claude Code's auto-memory: it is shared
by every worktree of the checkout, so it would carry unrelated work and earlier
runs into this one. What a run learns lives in its workspace.

Rerunning the same command resumes an interrupted run. `--clean` starts over:
it deletes the workspace's run state and every task worktree and `node/*`
branch in the checkout.

## The loop

```
plan   planner writes the first batch of plan.md — the tasks it can define
       before reading everything — and planner_turns/turn_0.yaml (errors are
       sent back until both validate)
run    every task whose dependencies are DONE starts, up to --max-parallel;
       each starts from the merged code of the tasks it depends on, and its
       changes are committed to branch node/<id>
       while the planner writes `planning: true`, it gets its next turn as
       soon as one is applied and adds the next batch; otherwise, whenever
       workers finish, the planner gets a turn (the others keep running),
       edits plan.md and writes planner_turns/turn_<n>.yaml:
         decision: CONTINUE | FINISH | ABORT
         retry:    [FAILED tasks to run again unchanged]
         cancel:   [RUNNING tasks to stop]
         planning: true while the plan is not complete (not with FINISH)
wrap   the final task is DONE: running side tasks get --side-wait-minutes
       (default 60) to finish, side tasks not started never start, and the
       planner may still chain another landing after the final task
done   nothing is left to run (succeeded), or the planner aborted
```

What a turn may change depends on each task's state:

| state | the planner may |
|---|---|
| `DONE` | nothing but its `priority` and `side` — its graph entry and `###` section are fixed |
| `RUNNING` | nothing, unless it `cancel`s it: it then starts over, or goes away if removed |
| `FAILED` | rewrite it, `retry` it, or remove it; until then its dependents wait |
| `PENDING` | anything |

A `FAILED` task holds back only the tasks that depend on it; nothing pauses.
A shared task does not start until the planner decides `FINISH`, so it can
keep adding work. A cancelled shared task's changes to the checkout, commits
included, are undone before it starts again. A rewritten, retried or cancelled task starts on a fresh
branch; its earlier results are kept as `task_result_<id>.attempt<k>.md` for
the next attempt and the planner to read. A turn that still does not validate
after the errors are sent back three times is discarded (`plan.md` is
restored) and the planner is told on its next turn.

A planning turn must add or change tasks. A discarded turn ends planning: the
planner's next turn waits for a worker to finish, and may write
`planning: true` again.

There is no turn limit. When nothing is running and nothing can start, the
planner gets a stalled turn; three in a row that start nothing stop the run
with an error (rerun to resume). After an interruption, rerunning resumes:
tasks that were running restart in a new session on their branch, with their
committed work, notes and detached jobs, and the planner is told.

## Steering a running run

`--feedback TEXT` (or `--feedback-file PATH`) only writes the message to
`<workspace>/inbox/` and exits; the running orchestrator notices it within
seconds and wakes the planner, whose next turn shows it first and is told to
act on it before anything else. The message moves to `inbox/consumed/` once a
turn that saw it has been applied, so a discarded turn or a crash hands it to
the planner again.

## The op primer

Every worker that writes or speeds up an entry starts by reading the same
things: the README, the `OpWrapper` contract, a few entries and their tests,
how GPU tests run here. On the Claude backend, a **primer** session reads them
once, in the checkout, while the planner writes its first batch, and writes a
summary to `knowledge/op.md`. Each `new_op` or `perf` worker then starts as a
fork of the primer's session — with that reading already in its history, in
its own worktree, under its own system prompt — and is told at which commit
the reading was done, to check what changed since before relying on it.

A task that is ready before the primer waits for it up to 20 minutes. If the
primer is not ready by then, failed, or a fork cannot start, the worker starts
afresh and reads `knowledge/op.md` first when it exists. The primer's session
is recorded in `primers.json` and reused when the run resumes; `--no-primer`
(and the Codex backend, which cannot fork sessions) starts every worker afresh.
`--clean` also deletes the primer's and the workers' Claude Code session
files.

## GPU pool

With `slurm-environment.gpu_pool` in `task.yaml` (see `task.example.yaml`),
the run holds whole GPU nodes for its workers. The orchestrator writes a
holder job (`gpu_pool/holder.sbatch`, which only sleeps) and keeps `nodes` of
them submitted from the moment it starts, so they queue while the planner
plans; it cancels them when the run ends. Each planner turn shows the pool
(nodes, free GPUs per node, waiting commands), and the planner may grow it by
writing `gpu_nodes` in its turn file, up to `max_nodes` (default: the number
of workers); the pool never shrinks during a run. An interrupted run leaves them
running for the resumed run (`scancel -n <job name>` releases them), and
`--clean` cancels them.

Workers run every GPU command through

```bash
python -m agent_flow.workflows.modeling_v2_onboard.gpu_run --pool <workspace>/gpu_pool \
    --task <id> --gpus N --minutes M <worktree> <command...>
```

which waits for N free GPUs on one holder job with at least M more minutes
(plus a margin), locks them, and runs the command as a step of that job in the
container, in the worktree, with `CUDA_VISIBLE_DEVICES` set. Waiting commands
are served by the task's class in `gpu_priorities.json` (`accuracy`, then
`feature`, then `perf`; first come first served within one), node by node:
a waiter ahead that fits nowhere claims the node with the most free GPUs, so
one-GPU commands cannot keep a node from ever draining for a whole-node one.
`perf` work leaves the last `reserve` free GPUs alone. Preparing the environment inside
the container (linking a built checkout's native libraries, the venv, model
paths) is work for the planner's first batch: a worker writes the wrapper the
other tasks then run their commands through.

## Side tasks

A `perf` node marked `side: true` is work the run does not merge: no task but
another side task may depend on it, the final task does not wait for it, and
it only takes the slots and GPUs the main line leaves free (its priority is
`perf`). The planner marks the unmerged `perf` tasks `side` once every goal in
`task.yaml` is met, so the run can land without another round of merging and
verifying, and keeps filling free slots with new side tasks until it lands.
`side` can be set and dropped on a task in any state without re-running it.

When a human asks to keep optimizing, the planner drops `side` and merges the
work: before the final task if it has not run, or — once it is `DONE` — by
chaining a merge task, an accuracy gate and a new `isolation: shared` task
after it, which lands the result and becomes the new final task.

When a run lands, each side task's branch is kept as
`<landed branch>-unmerged/<id>`, and `UNMERGED_OPTIMIZATIONS.md` is committed
to the checkout with each side task's status, result summary, base and task
description.

## Task tags

| tag | the worker is briefed to |
|---|---|
| `new_op` | write one new `OpWrapper` entry and its GPU test, certified on this machine |
| `target` | write or change a target: `modeling.py`, `weights.py`, routing, and its boot gate |
| `merge` | join several upstream branches, resolve conflicts, prove the join works |
| `module_accuracy` | check one module (attention, MoE, ...) against a reference |
| `e2e_accuracy` | run or debug the whole-model accuracy gate against the anchor |
| `perf` | make one entry, or one fusion of entries, measurably faster without changing its math |

A node's `type` is its primary tag; `tags: [...]` adds more. Each tag adds a
block to the worker's system prompt (`prompts/tags.py`).

A node's optional `priority` (`accuracy` > `feature` > `perf`) is its claim on
worker slots and shared GPUs: a free slot goes to the ready task of the most
urgent class (plan order breaks ties), and the orchestrator publishes the
classes in `gpu_priorities.json` for a GPU pool to serve. A running task is
never stopped to make room. Left out, the first `e2e_accuracy` gates built on a `target`
task (those checking the earliest code) and everything upstream of them are
`accuracy`, other `perf`-tagged tasks are `perf`, and the rest are `feature`. Changing it never re-runs a task.

## Plan rules

The planner's plan is checked before it runs (`plan.py`):

- every task has a `### <id>` section and a node in a flat graph;
- tags are from the table above; `kind: merge` exactly on `merge` tasks;
- a task that depends on two or more tasks is a `merge` task;
- exactly one final task, which every task but the side ones leads to, takes
  `isolation: shared` and the `merge` tag: it runs in the checkout itself and
  its commits are what the run delivers;
- other shared tasks are `merge` tasks that land one after another: each
  depends, directly or not, on the one before;
- `side` only on `perf` tasks that are not `merge` tasks, and no task outside
  `side` depends on one;
- on a revision, a `DONE` task's graph entry may not change or disappear.

## Workspace

| path | what |
|---|---|
| `task.yaml` | copy of the brief |
| `plan.md` | the current plan |
| `issues.md` | suspected bugs outside this model's delivery, recorded by the planner and not investigated |
| `task_results/` | `task_result_<id>.md` per task, plus earlier attempts |
| `planner_turns/` | `turn_<n>.yaml` per planner turn |
| `inbox/`, `inbox/consumed/` | human messages waiting for the planner, and those it acted on |
| `nodes/<id>/` | a task's notes and `jobs.json` (detached jobs) |
| `worktrees/<id>/` | a running task's git worktree |
| `test_commands.md` | Slurm commands the workers verified (Slurm runs only) |
| `primers.json`, `knowledge/op.md` | the op primer's session and the summary it wrote |
| `gpu_pool/` | the GPU pool's `pool.json`, holder script, GPU locks, wait queue and holder logs |
| `gpu_priorities.json` | each task's GPU priority class, rewritten whenever the plan changes |
| `.state.json`, `.running_plan.md` | task states and the plan being run, for resume |

## Requirements on the checkout

`trtllm_repo_path` must be a clean git checkout (no modified or untracked
files) and the workspace must be outside it: the final task commits
everything it finds there. Use a dedicated clone or `git worktree add`. On
success every `node/*` branch is deleted, except the side tasks', which are
renamed as above; when the planner aborts the run, the branches are kept so
unfinished work can be inspected.
