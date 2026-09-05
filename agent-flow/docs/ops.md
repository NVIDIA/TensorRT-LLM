# Operations tooling (`agent_flow.ops`)

Coordinator-side tools for a long autonomous run: reserving shared machines,
getting a running agent's attention, executing commands in a container without
burning the allocation, and watching the whole thing on a dashboard. None of it
is part of a workflow graph — a workflow can finish a turn without any of it —
but a multi-day run with several agents on shared hardware needs all of it.

Everything is driven by one config; see
[`agent-flow-ops.example.toml`](../agent-flow-ops.example.toml) (per project)
and [`agent-flow-ops.shared.example.toml`](../agent-flow-ops.shared.example.toml)
(per machine). Each tool takes `--config` and `--shared-config` and refuses to
run without one, printing every path it tried.

## Shared versus project config

The split follows ownership, not convenience:

| Shared, per machine | Per project |
| --- | --- |
| allocations (the reservation table) | project root, workspace, log dir |
| worktree slots | roles and their checkouts |
| container name, image, mounts, repo | notice channels |
| dispatch spool root | dashboard options |

Two projects on one machine must see ONE allocation table. A per-project copy
drifts, and two agents each believe they hold the same machine — which shows up
as an unexplained out-of-memory failure hours later, not as a conflict. A
single file containing both halves also loads, which is the right thing for a
one-machine, one-project setup.

## The tools

| Tool | For |
| --- | --- |
| `ops.tray` | claim / wait / release a shared machine allocation |
| `ops.worktree` | claim / release a pre-created git worktree slot |
| `ops.notify_agent`, `ops.ack_notice` | send and acknowledge a notice |
| `ops.in_container` | run one command in the allocation's container (one step) |
| `ops.container_dispatch`, `ops.dispatch_client` | daemon + client that run many commands in ONE step |
| `ops.bg` | detached run with separate stdout/stderr and an exit code |
| `ops.idle_watch` | alert a role when an allocation sits RUNNING and unreserved |
| `ops.ledger` | parse the roles' verdict ledger into a per-gate scoreboard |
| `ops.project` | list projects, scaffold a new one |

## The step budget, and why there is a dispatcher

Every command run into a held allocation with `srun --overlap` is a scheduler
*step*, and an allocation stops accepting new ones after a few hundred of them.
A one-shot wrapper spends one step per command — including one-second GPU
probes. A day of polling therefore kills a healthy allocation, and it fails in
the least helpful way: the allocation is still RUNNING, so the dashboard is
green, but every new command fails to start.

Hence `container_dispatch`: it spends ONE step per (job id, ntasks) pair and
then executes an unbounded number of commands inside it, talking to clients
through a spool directory:

* the client writes a request file atomically and tails the result back,
* rank 0 claims a request by renaming it, which is both the claim and the
  multi-rank barrier (every rank executes requests in sequence order, so ranks
  cannot diverge and no cross-node lock is needed),
* each rank touches a heartbeat file every few seconds. That heartbeat is what
  makes the client safe: it never hangs on a dead daemon, and it never falls
  back to a per-command step, because falling back is exactly the behaviour
  that burns the budget. No live daemon means exit 5 and a printed `--start`
  command.

Requests run sequentially per daemon. A long run blocks the queue for its
duration; that is intentional, and the workaround is a second lane (a different
ntasks or allocation), not concurrency inside the daemon.

## The role-addressed notice protocol

A single "notice file" loses messages: a second notice overwrites a first one
nobody has read. The queue is append-only JSONL instead, one object per line,
with notices, acks, follow-ups and reports.

* A notice carries `to`: the list of roles it addresses. It stays pending for
  each addressee until THAT addressee acks it. A record with no `to` is read as
  addressed to everyone, and an ack with no role settles it for everyone —
  older records must keep meaning what they meant.
* Ids are minted under the lock and are unique across every record type. When
  only notices were counted, an agent-minted ack id posted before the matching
  notice existed pre-acknowledged the notice that later got that id, and the
  real ack was dropped as "nothing pending".
* `--later` acks now and promises a follow-up: the sender knows the message
  landed, and the notice reads "follow-up due" until the result is posted.
* `--block` is a hard gate: the container and background wrappers refuse to run
  anything (exit 4) until the notice is acknowledged. Use it when continuing
  would waste hours, not to demand attention.
* Delivery is best-effort and layered: the queue is the record, and the notice
  is mirrored into the command cache every role reads before running a command
  and into the live-notes file a hook injects mid-turn. Never inject a notice
  into a wrapped command's own output — that output is redirected into the
  run's evidence logs, and a banner once ended up inside a file the run cited
  as acceptance evidence.
* Who acked is recorded, not guessed: `--role`, else the cwd matched against
  the config's role-to-checkout mapping.

## Mailboxes

A notice addressed to a *role* is too narrow once more than two participants
exist. A **mailbox** is a name any participant can register; the workflow roles
are mailboxes that happen to be declared in `[roles]`, and the queue file, the
record types and the id space are the same, so old records keep working.

```
python -m agent_flow.ops.mailbox register oncall --kind human
python -m agent_flow.ops.mailbox send --to coder,reviewer --key j17 "switch to X"
python -m agent_flow.ops.mailbox recv --as coder
python -m agent_flow.ops.mailbox ack  --as coder --id n7 "switched"
python -m agent_flow.ops.mailbox fsck
python -m agent_flow.ops.mailbox nag
```

`notify_agent` and `ack_notice` are thin wrappers over this layer and keep
their flags.

* **Delivery is best-effort, the queue is the record.** A mailbox names
  delivery hooks — `command_cache` (prepend to the file every role reads before
  running a command), `live_notes` (append to the file a PostToolUse hook
  injects), `tool_result` (write a file the harness injects into the next tool
  result, optional by design). A hook that fails is reported and never changes
  whether the message is pending. Hooks that write one project-wide file run
  once per message, not once per addressee.
* **Sends are idempotent under `--key`.** A caller that cannot tell whether its
  first send landed retries with the same key and gets the original record
  back.
* **Acks are recorded by mailbox name**, with `ack_source` saying whether the
  name was explicit, from `$AGENT_NOTICE_ROLE`, or inferred from the cwd. The
  inference is a fallback only: a wrong guess silently settles someone else's
  message, so it is worth being able to see afterwards.
* **`fsck`** reports acks for a message that does not exist, acks stamped
  before the message they answer, and messages addressed to nobody or to an
  unregistered name. Exit 1 when anything is found.
* **`nag`** escalates one step per round over what is overdue — a message past
  `--due` (or `[mailboxes].default_due_minutes`) with an addressee that has not
  acked, or a promised follow-up that never arrived: re-deliver, then mark it
  OVERDUE for the dashboard, then post to the human mailbox.
* **`status()`** returns pending, blocking, overdue and awaiting-follow-up in
  one call, which is what the dashboard reads.

## Reservation etiquette

Allocations and worktree slots use the same table (JSON under `flock`, rendered
to markdown for humans — read the markdown, never edit either by hand).

* Claim BEFORE launching anything; release the moment the work ends.
* Keep holds short unless the purpose says otherwise, and write a purpose a
  stranger can act on ("who to ask, what would free it early").
* Do not take a slot someone else holds. `--force` exists for a human
  reclaiming a slot from a dead agent, and it is logged as forced.
* The table records who INTENDS to use a machine; the machine records who is
  actually using it. Check both before launching.
* Renaming an allocation in the config migrates a live table through its
  `aliases` list, so a rename never orphans a reservation. Slots that
  disappear from the config are kept, not dropped — dropping one would discard
  a reservation whose holder is still running.

## The dashboard: collector, then renderer

```
python -m agent_flow.ops.dashboard              # one project, plain text
python -m agent_flow.ops.dashboard --json       # the status dict itself
python -m agent_flow.ops.dashboard --projects   # cross-project index
python -m agent_flow.ops.project list           # the same index
```

`collector.collect(cfg)` returns a plain dict assembled from the verdict
ledger, the two reservation tables (read without their locks — a viewer must
never block a writer), the mailbox queue via one `mailbox.status()` call, and
whether a workflow process is alive. `dashboard.py` only formats that dict, so
a second renderer costs nothing and both are testable without a terminal. A
curses view can be layered on the same dict later.

Narration is optional in the strong sense: with no narrator configured, no
binary on PATH, a non-zero exit or a timeout, the field reads
`(narration unavailable)` and nothing else changes. A dashboard that cannot
render without a model call stops working the day the model does.

The `--projects` index gives one row per project: parent, start commit, final
commit once it has been archived, flow state (running / idle / archived), last
ledger row, scoreboard, allocations held and pending messages. Allocations are
attributed by holder — the project name, or a `<project>/<role>` holder, or a
role name no other indexed project declares, since a shared word like "coder"
would otherwise show a second project's allocation as this one's.

## Archive and frozen viewing

A finished project is archived by copying its run directory (logs, workspace,
ledger, evidence) somewhere immutable and pointing the dashboard at that copy
instead of a live root. In frozen mode the collector reads only the archived
files: no scheduler queries, no process probes, no narration calls. That makes
a finished run reviewable months later, on a machine with none of the original
infrastructure, which is the point — a run whose evidence only renders on the
cluster that produced it is not evidence.

## Freezing a finished run

```
python -m agent_flow.ops.archive freeze <name> [--source DIR] [--dest DIR]
python -m agent_flow.ops.archive list
```

`freeze` copies the readable half of a project into
`<dest>/run-<date>-<name>/`: the workspace, logs, notes, the text files at the
project root, the text documents under `handoff/`, and every evidence file
under `evidence_max_bytes`. Larger evidence is *listed* in
`EVIDENCE-MANIFEST.json` with its size, mtime, and original path rather than
copied — the bulk a long run leaves behind (source trees, checkpoints, profiler
output) is not what makes it reviewable. `MANIFEST.json` records the archived
run's scoreboard, taken from the ledger in the archive itself, plus the repo
HEAD and branch when `--repo` names a checkout.

Two properties matter more than the copying:

* **Idempotent.** Re-freezing the same name refreshes the folder in place:
  unchanged files are skipped, and the README index row between the
  `<!-- index -->` markers is amended, never duplicated.
* **Nothing oversized enters git.** Any archived file over `git_max_bytes`
  (5 MB by default) gets its own `.gitignore` entry and a row in
  `MANIFEST.json["oversize"]`. A large file removed from git history costs far
  more than one never committed.

Symlinks are never followed, in either direction — a link inside the project
can point at a filesystem the archive must not absorb, so links are counted and
skipped. Override any of the copy lists or thresholds under `[archive]` in the
config.

## Scaffolding a new project

```
python -m agent_flow.ops.project new <name>     # dirs + a project overlay
python -m agent_flow.ops.project list           # every project and its state
```

```
python -m agent_flow.ops.project new <name> --parent <project>
python -m agent_flow.ops.project new <name> --start-commit <sha>
python -m agent_flow.ops.project new <name> --no-parent
python -m agent_flow.ops.project check [<project dir>] [--checkout DIR]
```

`new` refuses to scaffold without a starting commit unless `--no-parent` is
passed, and that refusal is the point: a project with no recorded start cannot
say later which of its parent's verdicts its code still satisfies, and "it was
green last week" is not evidence. `--parent` takes the final commit from the
parent's archive manifest (the archive is what froze; the parent's live
checkout has moved on), writes `parent` and `start_commit` into the project
overlay, and puts the GIT rule into the scaffolded `workspace/TASK.md`: branch
from that commit, the parent's ledger is frozen evidence, and a change touching
the code path behind a parent gate reopens that gate and only that gate.

`check` compares a checkout against the pin and prints the drift — `at`,
`descends` (with commits ahead and behind), `diverged` (the pin is not an
ancestor of HEAD, so the parent's verdicts may not apply), or `unknown-commit`
(the pin is not in this checkout at all). Exit 0 for `at` / `descends` /
`no-pin`, 1 otherwise.

`new` creates the project root with `workspace/`, `logs/` and `evidence/`, and
writes an `agent-flow-ops.toml` overlay naming the project and its roles. Fill
in the role checkouts, drop the task file into the workspace, and point the
tools at it with `--config <project dir>`. `list` reads files only — no
scheduler, no network — and reports, per project, whether a workflow process is
alive, when the newest ledger row was written, and how many gates are passing.
