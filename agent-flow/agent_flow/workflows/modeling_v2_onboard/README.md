# modeling-v2-onboard

Bring up a new model, or a new target of an existing model, in ModelingV2
(`tensorrt_llm/_torch/_experimental/modeling_v2/`). Two roles:

- **planner** — reads the task, the checkpoint and the catalog, and writes
  `plan.md`: tagged tasks plus an `## Execution Graph` of their dependencies.
  It never writes code. It is called again whenever a task fails. It is told
  `--max-parallel` and shapes the graph to keep that many workers busy:
  independent work side by side, feature work alongside the accuracy gate,
  and `perf` tasks in the slots left over.
- **worker** — runs one task in its own git worktree and ends by writing
  `task_results/task_result_<id>.md`. Workers run in parallel.

```bash
modeling-v2-onboard --task task.yaml --workspace /scratch/me/qwen3-onboard
modeling-v2-onboard ... --backend codex          # every agent on Codex
modeling-v2-onboard ... --max-parallel 4         # at most 4 tasks at once
```

Rerunning the same command resumes an interrupted run. `--clean` starts over:
it deletes the workspace's run state and every task worktree and `node/*`
branch in the checkout.

## The loop

```
plan     planner writes plan.md (errors are sent back until it validates)
build    tasks run in parallel; each starts from the merged code of the tasks
         it depends on, and its changes are committed to branch node/<id>
         ├─ every task DONE  → the run succeeded
         └─ a task FAILED    → the run pauses (running workers are interrupted)
replan   planner reads every result and writes planner_decisions/round_<n>.yaml
         ├─ REVISED → the failed tasks, the changed tasks and everything
         │            downstream of them are reset; back to build
         ├─ DONE    → stop: the results already prove the completion criteria
         └─ ABORT   → stop: the goal cannot be reached
```

There is no round limit. Interrupted workers resume in a new session on the
same branch, with their committed work, notes and detached jobs. A reset task
starts on a fresh branch; its earlier results are kept as
`task_result_<id>.attempt<k>.md` for the next attempt and the planner to read.

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
| `planner_decisions/` | `round_<n>.yaml` per replan |
| `nodes/<id>/` | a task's notes and `jobs.json` (detached jobs) |
| `worktrees/<id>/` | a running task's git worktree |
| `test_commands.md` | Slurm commands the workers verified (Slurm runs only) |
| `.state.json`, `.graph_state.json`, `.built_plan.md` | run state for resume |

## Requirements on the checkout

`trtllm_repo_path` must be a clean git checkout (no modified or untracked
files) and the workspace must be outside it: the final task commits
everything it finds there. Use a dedicated clone or `git worktree add`. On
success every `node/*` branch is deleted; when the planner stops the run, the
branches are kept so unfinished work can be inspected.
