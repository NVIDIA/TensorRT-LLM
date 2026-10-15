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
- **worker** — runs one task in its own git worktree and ends by writing
  `task_results/task_result_<id>.md`. Workers run in parallel.

```bash
modeling-v2-onboard --task task.yaml --workspace /scratch/me/qwen3-onboard
modeling-v2-onboard ... --backend codex          # every agent on Codex
modeling-v2-onboard ... --max-parallel 4         # at most 4 workers at once
```

Rerunning the same command resumes an interrupted run. `--clean` starts over:
it deletes the workspace's run state and every task worktree and `node/*`
branch in the checkout.

## The loop

```
plan   planner writes plan.md (errors are sent back until it validates)
run    every task whose dependencies are DONE starts, up to --max-parallel;
       each starts from the merged code of the tasks it depends on, and its
       changes are committed to branch node/<id>
       whenever workers finish, the planner gets a turn (the others keep
       running), edits plan.md and writes planner_turns/turn_<n>.yaml:
         decision: CONTINUE | FINISH | ABORT
         retry:  [FAILED tasks to run again unchanged]
         cancel: [RUNNING tasks to stop]
done   the final task is DONE (succeeded), or the planner aborted
```

What a turn may change depends on each task's state:

| state | the planner may |
|---|---|
| `DONE` | nothing — its graph entry and `###` section are fixed |
| `RUNNING` | nothing, unless it `cancel`s it: it then starts over, or goes away if removed |
| `FAILED` | rewrite it, `retry` it, or remove it; until then its dependents wait |
| `PENDING` | anything |

A `FAILED` task holds back only the tasks that depend on it; nothing pauses.
The final task does not start until the planner decides `FINISH`, so it can
keep adding work. A rewritten, retried or cancelled task starts on a fresh
branch; its earlier results are kept as `task_result_<id>.attempt<k>.md` for
the next attempt and the planner to read. A turn that still does not validate
after the errors are sent back three times is discarded (`plan.md` is
restored) and the planner is told on its next turn.

There is no turn limit. When nothing is running and nothing can start, the
planner gets a stalled turn; three in a row that start nothing stop the run
with an error (rerun to resume). After an interruption, rerunning resumes:
tasks that were running restart in a new session on their branch, with their
committed work, notes and detached jobs, and the planner is told.

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

## Plan rules

The planner's plan is checked before it runs (`plan.py`):

- every task has a `### <id>` section and a node in a flat graph;
- tags are from the table above; `kind: merge` exactly on `merge` tasks;
- a task that depends on two or more tasks is a `merge` task;
- exactly one final task, which every other task leads to, takes
  `isolation: shared` and the `merge` tag: it runs in the checkout itself and
  its commits are what the run delivers;
- on a revision, a `DONE` task's graph entry may not change or disappear.

## Workspace

| path | what |
|---|---|
| `task.yaml` | copy of the brief |
| `plan.md` | the current plan |
| `task_results/` | `task_result_<id>.md` per task, plus earlier attempts |
| `planner_turns/` | `turn_<n>.yaml` per planner turn |
| `nodes/<id>/` | a task's notes and `jobs.json` (detached jobs) |
| `worktrees/<id>/` | a running task's git worktree |
| `test_commands.md` | Slurm commands the workers verified (Slurm runs only) |
| `.state.json`, `.running_plan.md` | task states and the plan being run, for resume |

## Requirements on the checkout

`trtllm_repo_path` must be a clean git checkout (no modified or untracked
files) and the workspace must be outside it: the final task commits
everything it finds there. Use a dedicated clone or `git worktree add`. On
success every `node/*` branch is deleted; when the planner aborts the run, the
branches are kept so unfinished work can be inspected.
