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

`new` creates the project root with `workspace/`, `logs/` and `evidence/`, and
writes an `agent-flow-ops.toml` overlay naming the project and its roles. Fill
in the role checkouts, drop the task file into the workspace, and point the
tools at it with `--config <project dir>`. `list` reads files only — no
scheduler, no network — and reports, per project, whether a workflow process is
alive, when the newest ledger row was written, and how many gates are passing.
