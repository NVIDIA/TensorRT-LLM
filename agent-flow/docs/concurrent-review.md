# Switching a live `agent-team` run to concurrent review

Reference for operators. The design of `--concurrent-review` (pipeline
depth, verdict handling, fallback-to-sequential rules, shared-file
splitting) lives in
[`agent_flow/workflows/agent_team/README.md`](../agent_flow/workflows/agent_team/README.md);
this page only covers turning it on for a run that is already going, and
the two checkout layouts.

## Two layouts

`--concurrent-review` freezes the Coder's checkout at a commit and lets the
Reviewer read that commit while the next Coder turn runs. Who moves depends
on the `review_checkout` task key:

- **Coder moves (default).** The Reviewer reads the main checkout, frozen
  at `refs/agent-flow/review/iter-<i>`; the Coder works in a git worktree
  branched from the snapshot and folds its commits back after the verdict.
  Best when the Coder's environment is cheap to recreate.
- **Reviewer moves (`review_checkout: <path>`, "in-place" mode).** The Coder
  keeps working in the main checkout exactly as in sequential mode, and the
  Reviewer gets its own worktree, which the orchestrator detaches at the
  snapshot commit before each review. Choose this when the Coder's checkout
  is expensive to reproduce — prebuilt native artifacts, a virtualenv, a
  container that mounts the path — or when the task file forbids the Coder
  from creating worktrees at all. The reviewer checkout must be able to run
  the code: point the reviewer's tooling at it (most setups export a repo
  path environment variable that the container wrapper honours) and mirror
  whatever build-artifact symlinks the main checkout has.

Contradictory instructions are the main hazard: if the task file tells the
Coder "never create a worktree, work in place" while the default layout
tells it to move, the Coder will keep editing the checkout the Reviewer is
reading, which is exactly what the snapshot exists to prevent. Pick one
layout and make the task file agree with it.

## Reserving worktree slots

Both layouts need a worktree slot. Creating one on a network filesystem can
take minutes, so slots are pre-created and reused rather than added per
iteration. Track them with a table under a file
lock rather than a hand-edited document, so two agents cannot claim the same
slot. Point the task's
`worktrees_dir` at the directory holding the slots.

## Switching a running sequential run

1. **Wait for an iteration boundary.** Read `stage` from the workspace
   checkpoint `.agent_team_state.json`; `reviewer` means the Coder turn just
   finished, which is the cheap place to stop. Stopping mid-reviewer is also
   correct — that review is re-run — it just wastes the partial turn.
2. **Stop the process** (SIGINT to the workflow; if it runs under a terminal
   multiplexer, send the interrupt to that pane). Confirm the process is
   gone before continuing.
3. **Commit everything in the Coder's checkout.** A dirty tree is the one
   condition that silently keeps the run on the sequential path, because a
   snapshot of a dirty tree would show the Reviewer code the Coder never
   produced.
4. **Restart the same command without `--clean`**, adding
   `--concurrent-review` (plus `--review-snapshot-repo <path>` if the task
   file does not already name the repo). Resume is the default: the
   checkpoint decides the stage and iteration, and no workspace file changes
   meaning when the flag is turned on, so the switch is safe at any
   iteration boundary.

### Verifying the switch took

- The log announces concurrent review for the iteration instead of a
  fallback warning.
- The snapshot ref `refs/agent-flow/review/iter-<i>` exists in the snapshot
  repo, and the checkpoint's `review_snapshot_commit` matches it.
- In "reviewer moves" mode, the reviewer worktree's `HEAD` equals
  `review_snapshot_commit`, and the Coder's checkout does not gain
  worktrees.
- The Reviewer writes `status-review-<i>.md`, not `status.md`.

### Rolling back

Stop at the next boundary and restart without `--concurrent-review`. The
extra checkpoint fields are additive and ignored by the sequential path, so
no cleanup is required; leftover snapshot refs and worktree slots are
harmless and can be released later.
