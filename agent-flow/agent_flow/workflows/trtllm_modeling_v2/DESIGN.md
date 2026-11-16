<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# trtllm-modeling-v2: how a run works

The flow's design, for whoever changes it or wants to know why a run did what it did. A first
run needs none of it: [README.md](README.md).

## Terms

| Term | Meaning |
|---|---|
| Target | Model implementation for a checkpoint family and supported deployment configurations |
| Deployment | One served configuration of the checkpoint: native options, or `ctx` and `gen` each with their own. The first one defines `onboarded` |
| Session | One load of a deployment (one per role instance), with every check of it a client: `checks/session.py` |
| Unit, member | What the run's records (the ledger, a decision file's name) still call a deployment, by its name, and its shape (`tp4`, `tep4`, `dep4`, `pd6`). The spec does not use the words |
| Row | One piece of work the survey names: `operator` (an entry to certify or write), `weights`, `routing`, `state`, `feature`. Code makes one task per row |
| Card | What one worker's turn is asked, rendered from its row: outcome, what is required and optional, what to search for, reference code to look for, write scope, one check, the receipt's form. Under 600 words, and no place in the engine |
| Entry | A kernel wrapper with executable input, architecture and correctness contracts |
| Receipt | A task's recorded decision (`ready` or `blocked`), commits, its check as run, the references it read and what it left |
| Anchor | Two things, never confused in a message: an *accuracy anchor* is the score a benchmark is held to; a *tree anchor* is something the flow reads off the engine, found by its definition (`tree_map.md`) |
| Compute session | A process on a compute node that watches a spool and runs its queued commands |

## A run, step by step

1. **Specify and prepare:** derive or validate `task.yaml`, check the environment and agent access,
   survey available operators, and compile a contract of required checks: one session per
   deployment, and the native tests a deployment names.
2. **Tasks by code:** one survey turn writes typed rows (`operator`, `weights`, `routing`,
   `state`, `feature`); code makes one task per row and each deployment's own (its assembly then
   its session; for a disaggregated one its state, both sides, then the one session) and commits
   the package skeleton before they start side by side. No agent plans. A worker's card is
   rendered from its row; its receipt says `ready` or `blocked`.
   Each worker has its own whole tree of the run's branch. Kernel tasks certify or add
   operators; assemblers build the target. Commits are audited and integrated. A change to
   shared code that lands in the middle of a run starts no job: the tests it reaches run once,
   in the final pass, on the frozen candidate (step 3). A kernel task on a shared
   catalog entry lands a local variant in the target's own `ops/`, and the target is delivered
   with it: the run never changes a shared entry, so other models are not disturbed mid-run.
3. **Prove and deliver:** checks measure pins (one commit, with the build record of its
   libraries) against framework-owned references. A deployment is loaded once per session (see
   below) and every check of it is a client.
   Each result is a row of the ledger under a key: what the check measured, by content. A row
   is a check's result wherever the key is exactly the same, and nowhere else. Nothing is
   verified a second time in the middle of a run: a row never starts work. When every task is
   done or skipped the target is frozen, and one final pass runs on that candidate: each
   requested check must have a row under the candidate's key, so a check whose key is unchanged
   is not measured again and one whose key changed is, once. Then one selection of tests
   runs on the candidate, once: the target's own, those the tasks and the delivery name, those
   the change itself changed, and the package's collected suite when a change to shared code
   has an unknown reach. Its jobs state the devices the tests state, never a whole node by
   default, and each failing case is compared with the commit the run started from: one that
   fails there too is reported as such and fails nothing. A check or a test that fails there
   blocks the run (`RUN_BLOCKED`); the repair is a new candidate.
   Delivery requires the requested checks, the tests and the files to agree, and the closing
   report says for each requested check what it found and under which key.
4. **Resume:** checkpoints, decision identities and task receipts recover completed work without
   adopting another attempt's answer. A check that has its row in the ledger is not asked for
   again, and a job the last process queued and never read (still queued, running, or finished)
   is taken by the request that made it, not queued a second time.
5. **How a run ends** is the README's table. `onboarded` is pinned once, when the first
   deployment's session passes, and the closing report says under which requirements. A model
   whose load is long reaches that milestone first, in a run of its own
   ([spec/INTAKE.md](spec/INTAKE.md), section 3).

## Agents

Roles run on the backend, model and reasoning effort the spec states
([spec/INTAKE.md](spec/INTAKE.md), section 7). Roles load the run's own skills only:
preflight's agent probe reads the skills a session would load and turns off the user's, and Codex
roles run without the user's plugins and apps. Both backends use their native CLI's login,
configuration and state directories from the launching environment. The workflow never
copies credentials, parses native state configuration or chooses storage locations. It leaves
sandbox and approval policy to the native CLI and never upgrades the runtime on PATH.

## Compute

The run holds its GPUs in a pool and every GPU job is a file in one spool
(`<workspace>/spool` unless the spec names it). A *holder* is an allocation, or this one host; a
watcher on each of its nodes starts a job as its child within seconds, so small jobs never wait for
an allocation of their own. A job states its size (`gpus` or `nodes`) and its `wall`; one that does
not is refused in one line. Which environment gets the holders:

| `environment` | Who gets and ends holders |
|---|---|
| `slurm: {...}` (what a spec that names no environment is, where the host has `sbatch`) | the flow: it renders and submits a batch job per holder, renews it before the last one stops fitting the longest job, and ends it when no work is left |
| `kind: local` (what a spec that names no environment is, where the host has no `sbatch`) | the flow starts one watcher on this host and stops it at exit; the spool gives parallel jobs distinct devices. The kind is written into the spec the run reads (`<workspace>/task.yaml`), and the start's report names the devices it claimed |

A spec that states `environment.slurm` on a host without `sbatch` stops at its start and names
the alternatives; it never falls back to this host silently. A job wider than one node (`nodes=N`) is run once by the holder's
lead, which starts each rank world across the reserved nodes with one step line
(`infra/compute/slurm.py`). One orchestrator per workspace (a new one resumes from it), on a host
that sees the run directory, checkout and checkpoints at the same paths as every node.

The flow submits no job for its controller: it runs where the command is started. One that
finds itself inside a Slurm job stops starting turns ten minutes before that job's end, prints the
command that continues and exits 75; a turn that was cut runs again as the attempt it was. A site
that wants its controller in jobs of its own submits a script that runs the command and submits
itself again when the command exits 75.
A run on Slurm asks its scheduler for at most `environment.slurm.max_node_hours` node-hours (48
unless the spec states another number): every holder counts its nodes times its `time` from the
moment it is asked for, and once it is over, what it held. At the bound no holder is asked
for, what runs finishes, and once no holder is left the run exits 76.

The pool's four events and what answers each are [OPERATING.md](OPERATING.md)'s.
`trtllm-modeling-v2 pool status --workspace W` shows the holders and the submission records, and
`pool stop` ends them. The first start of a workspace runs the allocation probe by itself, for
the widest shape the spec needs, with no model loaded: the launch, the import, a collective
across the ranks, the cache directories and the dataset, in seconds. A pass is kept: a later
start asks for that shape again only when the spec's widest deployment changed. A session or test job the run has
waited on for 15 minutes writes a `waiting` event: the job, its spool, queued or running and since
when, why the watcher holds it when its `status.json` says, and for a queued job what runs on its
spool meanwhile; then at most one every 15 minutes while the wait lasts.

## Trees

The run asks four things of its trees (`run/workspace/integration.py`): `open(task)` a
writable tree at the run's head, `pin(task)` commit what is there and return the pin, `at(pin)`
that pin as a read-only tree, `land(task)` audit, rebase and fast-forward under one lock. **A job
runs a pin, never a moving tree**: a tree named at submission (`spoolctl --tree`, `gate --tree`) is
pinned then, and the watcher makes the pin's tree on its node before the job starts. A pin is a
source commit plus a build record, the content digest of each native library: a commit does not
carry the libraries git ignores, so where a job runs a pin the libraries there are the record's,
or the job has no result (`no_origin`).

They are whole trees made with git, **on each node's own disk**: a task's tree
on the controller's host, a pin's tree on the node its job runs on, under one path that is the
same on every node (`/tmp/mv2flow-trees-<user>/<run>`; `environment.trees.root` states another
place). A tree has a git directory of its own and the repository is told nothing of it, so the
storage every node shares gains the commits' objects and refs and no file of any tree. A pin's
tree is made once per node and used again; it is removed when nobody holds it and it went
unused for 30 minutes. A task's
tree that is gone with its node is made again from the task's branch and its last pin (a pin
is taken each time a job is queued against the tree), and the worker is told that the edits
made after that pin are lost.

Every tree links the checkout's one native build, and nothing inside a run rebuilds it. Work
that needs native code changed ends `blocked` and says what must change; the session that
watches the run rebuilds the base, and the run goes on from there. A worker reads this in its
turn's prompt, and a landing answers a branch that changed native source with the same text
(`NATIVE_BASE` in `run/workspace/integration.py`).
There is one kind of tree.

## Paths and reference code

The engine's code keeps moving, so the flow remembers no place in it.

- **What the flow reads off the engine is found by definition** (`operators/tree_anchors.py`):
  the attention backends by the classes on `AttentionBackend`, the custom ops by their
  registrations, the KV-cache dtypes by the assignment the loader makes, the accuracy tests by
  their harness class, the ModelingV2 guide by its title, and so on. `tree_map.md` in
  the workspace is written at each start: what each definition found, at which commit, and the
  search that finds it again. A definition no file holds is reported missing at the start; it
  is never replaced by a remembered path.
- **What a worker is told** is behaviour, names to search for as hypotheses, and reference code
  to look for with the reason it may fit: a lead, not a verdict. The survey's dispatch and
  every row's card name the engine's own model of the architecture first, found by its
  registration (`prior_art.json` in the workspace is what the run kept of it); what else fits
  a row the surveyor writes into the row's leads. A location a worker resolves is evidence about one commit and belongs in its receipt.
- **What a target may call and what it may only read or copy** is the engine's rule (the
  ModelingV2 guide, "Modeling V1 code"): a target does not import, subclass or call Modeling
  V1 model code, weight loading included; it may read that code and copy or adapt it. Every
  role reads the rule once, in its prompt. What the rule covers is read from the guide's own
  sentence and from what registers a model (`operators/v1_rule.py`), never from a remembered
  path.
- **The one place named** is the subject: the ModelingV2 package a target is added to and the
  directory its tests mirror it in (`operators/coverage.py`, `SUBJECT_PATHS`). A task's write
  scope is under them.
- **Where a run may commit** is not a list of the engine's directories either. Each task
  commits inside its own write scope, which the landing audits, and the watching session
  widens one task's with `allow-shared`; a change to code other models run is listed in the
  report and followed by the final tests. A spec that wants a fence states
  `commit_policy.allowed_paths`.

## One session per deployment

A deployment's checks share one load: one job that holds every role instance's devices and
states its wall; inside it one `trtllm-serve` per role instance on devices of its own (and
`trtllm-serve disaggregated` over `ctx` and `gen`); then the checks, each a client with its own
line in the ledger. Which checks run, in which order and when is
[spec/INTAKE.md](spec/INTAKE.md), section 4. `trtllm-modeling-v2 gate deployment --task
task.yaml --deployment <name>` runs one session as the run does.

`not verified` is a third outcome beside pass and fail: where the engine offers no way to
observe what a check is for (that generation received the context's state, say), the line says
so with the reason. It fails nothing and certifies nothing: the ledger's line carries
`unverified`, the contract does not hold with it (`partial`), and `gate deployment` prints
`status: partial` and exits 3 (0 pass, 1 fail or blocked, 2 usage or spec error). Nothing in
the tree repairs it.

A deployment wider than one node is one job of whole nodes (`nodes=M`) whose script the holder's
lead runs once (`checks/gang.py`): every rank first says where it is and which devices it was given,
and a device given twice ends the job before any load; then one step per role instance, built by
the pool's step line, each world on its slice of the reserved hosts (a server that fits in what
is left of a node shares it, one wider than a node takes whole nodes); then one task that starts
the proxy and runs the checks as clients. It needs `environment.slurm`; on one host such a
deployment is refused before any load, with that reason, as is a world wider than a node whose
ranks are no whole number of nodes. Every session job states its devices or nodes, its wall
and the ranks of each role instance (`ranks=ctx0:1,gen0:4`): a rank that reports no origin leaves
the job with no result.

## Code map

| Location | Responsibility |
|---|---|
| `cli.py`, `spec/` | Entry point; task schema, deployments, derivation and advice |
| `agents/` | Role/model routing, turns, the receipt's two decisions, prompts, the card (`prompts/card.py`), what counts as an engine path in a worker's text (`prompts/paths.py`) and the entry skill |
| `operators/` | What the shared catalog certifies here, operator coverage (`vocabulary.yaml`), typed rows (`work.yaml`), the engine's own model and limits, and the tree anchors |
| `run/workflow.py`, `orchestrator.py`, `graph.py`, `escalation.py`, `resolve.py` | Lifecycle; tasks by code and scheduling; graph rules; blocks, answers and the `resolve` verbs |
| `run/startup/` | Bootstrap, preflight and agent probes |
| `run/execution/` | Task dispatch, kernel-specific preparation and decision/receipt handoffs |
| `run/validation/` | The contract, the check runner, the final tests and completion |
| `run/workspace/`, `run/records/` | The run's trees (open, pin, at, land) and what one check measured; recovery state, the journal and the closing report |
| `checks/`, `checks/harness/` | A deployment's session, attestation, anchors and verdict rules; programs executed in the measured environment |
| `infra/compute/`, `infra/monitor/` | Execution, sessions, scheduler and provenance; pins and build records (`pintree.py`); status and heartbeat |
| `infra/git.py`, `infra/files.py` | Git questions and the commit rules; atomic file I/O |
