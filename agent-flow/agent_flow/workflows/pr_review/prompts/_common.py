"""Shared prompt boilerplate for the pr-review reviewer and coder roles.

``COMMON_RULES`` carries the two hard rules from the user (never post to the
PR/MR; the coder may push back and keeps the right to decline changes), the
workspace file map, and how to view the change. Both role prompts embed it so
the rules are stated identically to both sides.
"""

COMMON_RULES = """\
## Workspace

You and the other agent communicate through shared files. **The entire review
conversation stays local — it is never posted to the PR/MR.**

- `pr_context.md` — The PR/MR under review: identifier, title, description,
  base/head branches, the local repo path, and the **exact diff command** to
  run. Read this first.
- `progress.yaml` — The append-only audit log of the review. You never edit it
  directly; you append your turn via the `append_*_progress` tool below, and
  read the latest entries via `read_latest_progress`.
- `discussion.md` — The rolling, local review conversation: open threads (with
  status), points both sides agreed, declined push-backs (with rationale), and
  pointers for the next turn. Read it via `read_discussion` at the start of
  your turn and overwrite it via `update_discussion` before you finish. **Do
  not edit it with `Write`/`Edit`** — only via `update_discussion`.

## Seeing the change

Work inside the local repo named in `pr_context.md`. View the change with the
diff command from `pr_context.md` — a merge-base diff that **includes the
coder's uncommitted working-tree edits**, so it always reflects the current
state, not just the original PR/MR. Read the changed files directly, and
build / run / test as appropriate; do not reason from the diff alone.

## Hard rule 1 — never touch the PR/MR remotely

The conversation is **local only**. You must NOT post anything to the PR/MR or
mutate it on the platform. Forbidden commands include (non-exhaustive):
`gh pr comment`, `gh pr review`, `gh pr edit`, `gh pr merge`, `gh api` writes,
`glab mr note`, `glab mr approve`, `glab mr update`, `glab mr merge`. Use only
read operations against the platform if you need them (you usually don't).

## Hard rule 2 — leave edits in the working tree

The workflow never commits or pushes. Do NOT run `git commit`, `git push`, or
otherwise create commits — leave all edits as uncommitted changes in the
working tree for the human to inspect and commit themselves. (Read-only `git`
such as `git diff`, `git status`, `git log` is fine.)
"""
