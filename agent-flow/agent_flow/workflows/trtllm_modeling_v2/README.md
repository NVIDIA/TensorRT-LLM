# trtllm-modeling-v2

Brings one HuggingFace checkpoint to a gated **ModelingV2** target: a flat,
self-contained forward assembled only from catalog entries, trusted through
accuracy gates rather than through shared abstractions.

This workflow is the agent half of modeling_v2.
The code half is already in tree at
`tensorrt_llm/_torch/_experimental/modeling_v2/` — the catalog, the targets,
the router and their tests. What lives here is what builds them: the
orchestrator, the agent roles, and the skills those agents load.

```bash
trtllm-modeling-v2 --task task.yaml --workspace workspace/trtllm-modeling-v2
```

Or let the run derive its task from the checkpoint (the fast path,
[INTAKE.md](INTAKE.md#the-fast-path)): the device, sessions and defaults come
from the cluster's `site.yaml`, the features the checkpoint carries are built
unless the goal leaves them out, and the request is written to
`<workspace>/task.intake.yaml` and run at once. Its intake checks run in phase
0 and stop the run with every block listed; a confirmation is recorded
(`--confirm`), not awaited.

```bash
trtllm-modeling-v2 --checkpoint /models/newmodel --site site.yaml --workspace W \
    --goal "mtp disagg@tp1/tep4 tp1 tep4 dep4 isl=8192 osl=1024"
```

Start with `--dry-run`. It runs preflight, resolves the launch environment and
stops before the first agent turn — the cheapest way to find out whether this
host can host this run. `task.example.yaml` documents every key of the spec.

Three companions to this file: **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** is keyed on
symptoms — the verbatim error you saw. **[BUILD_AND_TEST.md](BUILD_AND_TEST.md)** is
keyed on tasks — building, rebuilding, and running at CPU / 1 GPU / multi-GPU /
multi-node scale. **[LESSONS.md](LESSONS.md)** is keyed on what real runs got
wrong — each lesson's symptom, its cause, and the mechanism in this workflow
that now prevents it, or the gap that is still open.

**When something breaks, read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) before
you debug the target.** Every entry there read as a model or target defect
while it was happening — `MPI_ERR_SPAWN`, a kvCache that cannot fit a
sequence, NVRTC missing `cuda.h`, a 27-point accuracy regression that was a
protocol mismatch — and every one was the environment.

## The pipeline

```
preflight  ->  planned run  ->  delivery
  Python       planner + workers, checks run by the orchestrator        Python
               (a `tune` task joins once the core is correct)
                        |
                    improver: lessons, flow issues and patch drafts, beside the run
```

| step | who | what it produces |
|---|---|---|
| `preflight` | **nobody**: pure Python | the slug, the branch, the resolved launch prefix, the device claim (local runs), the substrate gate, the surface survey, `contract.yaml` (every check the run owes) and `coverage.yaml` (what those checks prove of the request, feature by member; a requested cell left uncovered stops the run unless `coverage.accept` names it) |
| planned run | the planner, the workers, the orchestrator | the planner writes the task graph (`graph.yaml`); the orchestrator starts its tasks as capacity and their files allow, runs each finished task's checks itself, and wakes the planner when something happens. It ends when every required check passes on one tree |

The first two overlap (phase 0). Preflight's checks take seconds; the
substrate gate, its GPU boot, runs beside the planned run's first turns rather
than before them, and its failure stops the run there, which a re-run of the
same command resumes: a stopped run is left open, not closed, and wakes no
improver. A run started with `--checkpoint` runs its intake checks
first: the request's feasibility checks and the anchor, speculation and
capture-list checks of the spec compiled from it, every block reported at once
before the host is touched. A new run starts with the survey (task `survey`) beside
the planner's first turn, and when the survey lands the orchestrator adds one
kernel task per vocabulary queue row (`k<row>-<entry>`, touching the row's
entry and its test), which start at once; the planner wires the rest of the
graph to them and may reshape those not yet running. A row on a shared entry
touches its local variant in the target, not the catalog file (see
[Develop in the target, then promote](#develop-in-the-target-then-promote)).
A shared entry has one row, every call site's shapes as its cells.
| delivery | Python | the gate findings, the trees on disk, and the contract on the target as it is |

The planner orders the work freely: nothing fixes stages, units or their
order. Each task has a kind, and each kind runs as one role:

| kind | role | does |
|---|---|---|
| `survey` | `plan_drafter` | maps the checkpoint's computation steps onto the catalog: `plan.md`, `vocabulary.yaml` |
| `kernel` | `smith` | one catalog entry, from a `vocabulary.yaml` queue row or `<mode>:<entry>`; or a `promote:<entry>[,<entry>...]` of local variants into the catalog |
| `assemble`, `feature`, `integrate`, `investigate` | `assembler` | the target's core, one feature on it, a merge of parallel branches, a diagnosis |
| `review` | `reviewer` (`plan_reviewer` for `ref: plan`) | reads a diff (or the plan) against its evidence, for what no check sees |
| `tune` | `tuner` turns under Python | the tune campaign; see [Tuning](#tuning) |
| `check` | none: the orchestrator | runs the contract checks in its `satisfies`, with no agent turn |

Each `kernel`, `assemble`, `feature` and `integrate` task works in its own
worktree (`worktrees/<task-id>`, branch `mv2/<slug>/<task-id>`) cut from the
checkout's head, and commits there through `bin/commit.sh`
(`MV2FLOW_COMMIT_TREE=<worktree>`). When it finishes, its branch is rebased
onto the head and fast-forwarded in (`run/integration.py`), so the checkout
only ever moves by whole tasks and carries no merge commits. A rebase that
conflicts goes back to the task's worker, up to twice, naming the files and
the head to rebase onto: the worker resolves in its worktree, re-runs its
tests and hands back, and its branch is integrated again. Only then does the
task end `INTEGRATION_FAILED`. An `investigate` task gets a scratch worktree
that is never integrated.

A task worktree is a sparse checkout: the package, the tests and examples a
worker runs, the directories its task `touches`, and the files at the top
level; a task that touches `cpp/` or `3rdparty/`, or runs `exclusive`, gets
the whole tree. The worker widens its own with the `git sparse-checkout add`
command its prompt gives. No config file is written for it: the cone is kept
in the worktree's own git directory, so a repository whose shared
`.git/config` is read-only or locked gets sparse worktrees too. Each
checkout writes 16 files at once (`checkout.workers`), and how each worktree
was made (sparse, or whole and why) is a journal `integration` event.

Ready tasks start together, whatever directories they share, unless both name
the very same file in `touches` (a path whose last part has a suffix) or
either is `exclusive: true`; an exclusive task that waits holds back every
task ranked after it, so the run drains to it. A `nice_to_have` task only
gets devices no waiting required task can use. Tasks are admitted by
capacity: the devices of the live spool schedulers (`status.json`,
quarantined devices left out) less the `gpus` the running tasks declare, or
the site's device count without a live session. An agent turn holds no
device, so gate jobs that fill the nodes keep no task waiting. A review, a
`verify` kernel row and a `check` task reserve none (`resources.gpus: 0`;
the orchestrator's own verify rows get none): their tests and gates are jobs
that take their own devices as each starts.

A `check` task is contract checks and nothing else (re-certifying a member
after a change lands, a stale check run again). Once its dependencies are
done the orchestrator sends its checks to the check pool itself, with no
agent turn; it takes no task slot, its `touches` are empty and its `gpus` 0.
It ends done when they pass, with a receipt naming them; only a failure
wakes the planner, which sends an agent (an `investigate` task) or retries it.

A task may soft-depend on another (`depends_on_soft`; both land on the
target): it starts once the other has started rather than once it is done,
and is told where each soft dependency stands. When its worker finishes, its
integration waits for each of them to land or end, so its branch is rebased
onto a head that holds what landed and its checks run there. One that ended
without landing is named in a line after the task's decision, which reaches
its receipt and the planner's event, and the planner hears at once which
tasks go on without it.

When the run binds the cross-model rule (a new run does; one with work done
before it keeps the rules it started under), every task that lands a change
outside the target's own paths needs a required `review` with
`regression: true` after it, and such reviews are batched. A regression
review covers, as it starts, every change to shared code that landed and no
earlier review covered: its batch, recorded as `covers` (`<task>@<attempt>`)
in `graph.yaml`, whatever its `depends_on`. While a task that changes shared
code runs, a ready regression review waits up to 30 minutes after its
batch's first landing for that change to join; one runs at a time, and what
lands meanwhile joins the next batch; one whose batch is empty when it would
start closes with no turn. A failed review covers nothing, a retry that
lands again is a new change, and so are the guard commits a failed receipt
lands. The planner's request says what the next batch holds, a refused
revision names the open review its new shared tasks should be added to, and
the run is not delivered while a landed change to shared code is in no
review's batch.

A kernel task's smith is dispatched with its task's goal beside the queue
row's surface and note, and with the sourcing tier on record for its entry:
its own row's, else the `tier` the last smith that landed the entry wrote in
its decision file, else the vocabulary step that maps onto the entry. With
none on record it is asked to name the tier, not told tier 1.

### Develop in the target, then promote

Every certification of a shared entry used to write the one catalog file, and
two tasks that declare one file never run at once, so certifications of one
entry ran one after another. A kernel task on a shared entry
(`<category>/<name>`) now lands a **local variant** in the target's own
`ops/`: `ops/<category>/<name>__<site>.py` and its test
`test_modeling_v2_<name>__<site>.py` beside the target's other op tests
(`vocabulary.entry_paths`). The site is the row's `site` (lowercase words
joined by single underscores), else `r<row>`; a `verify:<entry>@<site>` task
names it after `@`, else its task id does, and a task whose `touches` declare
a variant's files lands there. A variant starts as a copy of the entry,
keeps its class and instance name, and the target imports it. Variants touch
only the run's own paths, so no cross-model regression review follows them.
A task that declares the catalog file instead lands in place, as before.

Before delivery one `promote:<entry>[,<entry>...]` kernel task
(`onboard/promotion.py`) merges each entry's variants into the catalog: one
entry with the union of their cells, guards and `ARCHS`, one test file, the
target's imports pointed at the catalog, the variants removed, and the merged
tests with the claims gates run once in one process (the verify skill's
`references/promote.md`). `promotion.plan` adds one per entry as soon as it
is due, beside the feature work: no kernel task still to land a variant of it
is pending or running, and the target already imports each of its variants
(or nothing is left to change the target). Every entry due at once goes in
one promote. It declares every file it rewrites (the catalog entries and
tests, the variants, the target files that import them, `modeling.py`), so a
task declaring one of them never runs beside it. An entry with an open
promote waits for it; after one fails the planner retries or reshapes it. An entry `task.yaml` keeps
local (`target.keep_local`) is never promoted, and delivery accepts its
variants.

A task that finishes leaves a receipt, `tasks/<id>.receipt.yaml`
(`run/receipts.py`): its decision, the first paragraph of its summary, the
commits its branch integrated, a smith's entry and test result, the commit a
cross-model regression review found nothing regressed at (`cleared`), its
decision and result files, and the notes its worker addressed to the
orchestrator or the planner, each bounded. A task that starts carries the receipt of each
task it depends on in its envelope and its prompt, under `## Receipts of the
tasks this one depends on`, so no planner turn goes to restating them in a
goal.

A worker reports that it finished; it never reports that a check passed.
The orchestrator runs the checks a task claims (`satisfies`) once its branch
is in, all at once on one pool every task shares, as many at a time as the
sessions' free devices can start (at least 2, at most 32; 8 when no session
can be read), so one long gate never holds up the rest of the graph; the task keeps its
slot until its checks are in. A claimed delivery-tier check waits for the
delivery tier.

A check never measures the checkout, which integrations move while it runs.
It measures a snapshot of the commit the checkout is at when it starts: a
detached worktree of that commit under `<workspace>/snapshots/`, sparse with
a task worktree's cone, the checkout's native build linked in, its files
read-only. Every check of one commit shares that snapshot; a check leases it
while it runs, and once the checkout has moved past its commit a snapshot no
check holds is removed. Every command of the check runs from inside the
snapshot with it first on `PYTHONPATH` and, on a spool, names it as the job's
tree. The verdict names the commit and, in `contract_status.jsonl`, what each
command (each leg, each engine job) imported the package from, as the spool's
resolve hook and the engine driver print it. A verdict whose commands
imported more than one tree, or a checkout no commit pins, is `VOID`: it is
measured once more, and if it stays void it is recorded as not passing. A
verdict is recorded under the state of the target it measured, so one taken
before another task's integration is stale and runs again once nothing but a
tune campaign is running (it measures only its own worktrees), and a check
the contract already counts as passed on the current state is not run again.

That state is a fingerprint of the tree the gates measure: the commit the
checkout is at, less what no gate runs or reads
(`tests/`, `docs/`, `.github/`, `examples/`, `*.md` anywhere, and the run's
workspace), but with the files a gate does read there (the quickstart, the
default boot image, and every file in the checkout that the spec's
`accuracy`, `runtime`, `variants` or `environment` block names). Of an anchor
table a unit or variant names as `reference_yaml`, only the entries of the
model keys the run reads count, unless the spec names the table elsewhere
too. The native libraries built into `tensorrt_llm/` and `tensorrt_llm/libs/`
are outside git, so their sizes and modification times are part of it. A
commit that only edits a test list or another model's anchor leaves every
verdict standing; a commit that changes any code, or a native rebuild, makes
every verdict on the target stale. An uncommitted change in the checkout is in
no snapshot: it changes no verdict, and each check it would have reached
warns that it did not measure it. What a check reads from outside the
checkout (the spec as it applies to its unit, the options' content, an
anchor, a prompts file, the logit reference) is its `inputs` digest, recorded
on every verdict: a run relaunched with other inputs runs again the checks
they reach. Boot, accuracy and acceptance checks of different units that run
one command on one engine against one criterion (the same runtime block,
speculator, options content and workload; the same accuracy or acceptance
inputs) share a verdict: a pass of one on the current tree holds for the
others (`SHARED` in the standing). A failure is never shared.

The boot, logit, acceptance and accuracy checks of one unit at one member
that are ready together start one engine between them (`checks/bundle.py`).
They join a bundle as they are admitted: the first leads it and holds the
one admission slot the bundle takes, the others wait in it holding none.
After a short gathering window (2 s) the leader captures the command each
member's gate would launch, on one snapshot of the checkout's head, and the
quickstart commands whose launch, quickstart and argv agree (but for
`--max_tokens`) form one group. An accuracy check joins that group only when
its `trtllm-eval` argv and options file state every engine setting either
launch path can state the way the quickstart argv does (`max_batch_size`,
`max_seq_len`, the KV-cache settings, the capture list, the speculator, the
topology, the attention and MoE backends) and its options file sets nothing
else; the acceptance gate's `=off` leg and a check without the speculator are
other engines. The group runs as one job (`harness/bundle_driver.py`): the
engine is built once, then each member's workload runs in a section of its
own, the evaluation last, in process. Each gate then judges its own section
as the output of its own command, so every verdict is its gate's own, every
member's carries the one engine record they share, and the verdict names the
bundle (`[bundle <name>: <checks>]` in the journal, `bundle` in
`contract_status.jsonl`). A section that failed, ran past its own time or
never ran (and every section after it) runs again singly, admitted on its
own; so does an evaluation whose engine, once up, was not the one its options
file declares. When the engine could not be built, each quickstart gate takes
that failure as its own job's. The journal says which checks were judged from
each bundle and why any ran singly (`bundle` lines).

The planner's turn runs on a thread of its own, one turn at a time: tasks
keep finishing and starting from the current graph while it thinks, and its
revision is applied when the turn returns. A failed task or check, a worker's
report, a critique, a failed re-check, or nothing left to run or start wakes
it at once. A task that finished with its checks green waits up to
`--event-window` seconds while anything is in flight, so successes reach the
planner together. A green task that satisfies no check and that an open task
depends on wakes no turn at all: the graph already says what follows it, and
it waits for the next turn something else wakes. What arrives during a turn
goes to the next one, and stays
in `planner_events.json` until it is read, unless the revision that turn
wrote already answered it: a task it retried or dropped, or one it wrote
done at the attempt that finished. The first graph, and any revision that
changes more than half of the open tasks, gets one advisory read by the plan
reviewer (`--no-graph-critic` skips it) on a thread of its own: nothing waits
for it, and its words reach the planner's next turn.

A failed task runs again under its own id when the planner writes it back as
`pending`. Each such retry is the task's next attempt (`attempts` in
`graph.yaml`, the orchestrator's count), and a revision that retries a task
whose third attempt failed is refused. A retried kernel task's smith is
dispatched afresh rather than handed the failed attempt's decision, and a
retried tune task keeps its measurements apart (see [Tuning](#tuning)). A
retry works in its task's worktree, whose branch is first rebased onto the
checkout's head (`Integration.refresh`): it starts from what landed since,
with its earlier attempts' commits on top. A worktree with uncommitted
changes, or a branch that does not rebase cleanly, is left as it was, and
its worker is told how far behind the head it is. A kernel task that ends
`FAILED_RECEIPT` has its commits integrated when they change only the
entry's own files (the entry or its local variant, its test, a new
`ops/<category>/__init__.py`):
the refusal and its guard test reach the target whether or not the task runs
again; anything else stays on the branch for the next attempt. A
task the run's infrastructure failed (its turn raised, it ended
`INTEGRATION_FAILED`, or its tune baseline could not be measured) starts
again once by itself, as the same attempt, and the planner hears why; the
count is kept in `planner_events.json` (`retries`), so a restart grants no
second one.

Every step checkpoints, so a crash resumes where it was: the graph, the
verdicts and the tune campaign are on disk (`graph.yaml` and
`planner_events.json` are replaced whole, never half-written), and a task
that was running when the process died starts again. Re-running the same
command resumes; `--clean` starts over.

The bring-up comes first: the first state of the target on which the first
unit's boot and accuracy hold at its narrowest member (tp1) is named the same
way, once, with the branch `<target branch>-bringup`, the journal line
`BRINGUP DELIVERED <sha>` and a `bringup` line in `correctness.jsonl`. It is a
usable one-rank target hours before every member, feature and variant is
certified.

Correctness does not wait for performance. The tune campaign is background
work: no task depends on it, and its best branch is reported as an artifact of
the run, never integrated in it. The delivery tier runs once every other
required check holds and no required task is open but the campaign. Full
support is one predicate: every required check holds on the target as it is,
the delivery tier's included, the delivery files on disk check out, every
landed change to shared code has had its regression review, and no local
variant waits for promotion. The first state on which it holds is the
correctness milestone, recorded with no planner turn: the branch
`<target branch>-correct` (the branch checked out in `trtllm_repo_path`; the
run's own branch name when it is detached) is set to that commit, the journal says
`CORRECTNESS DELIVERED <sha>` with the hours since the run opened, and
`correctness.jsonl` gets a line. Later work that turns a check red or stale
leaves it where it is: each new failure on a later state is journaled as
`correctness stays delivered at <sha>: ...`, and the planner's standing and
the run's closing journal sections say whether the target has moved since.
The next state full support holds on moves it forward. Once it holds, a
running campaign is told to wrap up: it dispatches nothing new, drops what it
has not started measuring, and ends with its best kept tree; then the run is
delivered.

The run closes with one verdict: `delivered`, `tuned` (delivered, and the
tune task finished), `escalated` or `ceiling` (the planner's), `planner-stalled`
(`--revision-cap` revisions without a check turning green), `planner-refused`,
`planner-failed` (five planner turns in a row raised), or a `delivery-failed:*`.

## Why it is shaped this way

The predecessor workflow for this job reliably failed the same way: runs spent
their budget deciding *how to launch a job* and re-litigating a plan, and never
reached model onboarding. Four things here exist to prevent that, and they are
the parts worth not "simplifying" later.

**Preflight is Python, not an agent turn.** The checkpoint parses, the arch
matches the host, `TMPDIR` is sane, a local run's devices are claimed by
`memory.free` and a lone-process probe, a multi-rank run's NCCL/MPI substrate
answers, and the stock path generates a token under the run's runtime block —
all before any model is invoked.

**The launch environment is declared once and handed to every agent.**
`task.yaml`'s `environment` block resolves into a single command prefix that
every role's prompt carries verbatim. Agents never construct a launcher
command, never probe for one, and never call a test-runner skill. A prefix
that does not fit is a *symptom to report*, not a problem to solve.

**Rules constrain outcomes, not process.** The contract is the only
definition of done, and only the orchestrator runs it. How the work is split
and ordered is the planner's call; the planner cannot mark a check, relax a
requirement or write code.

**Gates are exit codes, not opinions.** There is no `--min-score` and no agent
scoring an artifact out of ten. `trtllm-eval` prints a number, the anchor comes
from TensorRT-LLM's own accuracy references, and the verdict is
`measured >= floor`.

## The gates

| gate | when | what |
|---|---|---|
| **boot** | inner tier: after every task that claims it | minutes. The declared boot workload through the engine driver; keywords are matched on generated text only, and a multimodal target is sent an image. With a logit reference, the reference's first text prompts go as token ids and the engine's first greedy tokens (4 prompts × 24 by default, `accuracy.boot.greedy`) must be the reference's, up to a near-tie no wider than the logit gate's largest \|Δ log-prob\| limit; a mismatch fails the boot before any benchmark runs |
| **accuracy** | inner tier: after every task that claims it, at every member, or with the logit gate on at the narrowest, the widest and every attention-DP member (and every member of a speculator unit whose boot compares no tokens); `accuracy.benchmark_members: every` keeps every member. `contract.yaml`'s `notes` say where and why | the release criterion. `trtllm-eval` against the anchor, with a statistical floor |
| **witness** | proof tier: each feature boot and accuracy cannot see, every member | two legs that differ in exactly one variable; see [Feature increments](#feature-increments) |
| **acceptance** | proof tier: every unit that keeps a speculator on, every member | `acceptance_length` on ≥ 32 natural-text prompts, paired against the previous verdict |
| **envelope** | delivery tier: once no required task but a tune campaign and what waits for it is open, every member | the finished target at its declared limits: `max_batch_size` in flight, prompts at `max_seq_len`, every capture batch replayed; probes must keep their first tokens from a batch-1 run (`checks/envelope_boot.py`) |
| **serve** | on wherever the spec bounds the server's sequence length (`accuracy.max_seq_len`, or `max_seq_len` in the run-wide options file); `enabled: false` turns it off. Inner tier: every unit, at its widest member; delivery tier: the last unit at one rank | `trtllm-serve` starts the target with the accuracy gate's options file and answers one `/v1/completions` request, then the boot workload's prompts the boot gate judges (each with keywords; an image prompt as a `/v1/chat/completions` request carrying its image); every answer must have text, and hold the keywords the boot gate holds the engine's to. The server and its ranks are stopped either way (`checks/serve_gate.py`) |
| **logits** | wherever `accuracy.logits.reference` names a file, or `logits.reference.json.gz` sits beside the spec; `enabled: false` turns it off. Inner tier: every unit, every member | the reference's fixed prompts through the boot gate's engine, speculator off, with prompt and generation log-probs; at every position whose target is not an image placeholder, the reference's top-1 token is compared: top-1 agreement (pooled), mean \|Δ log-prob\| (per prompt) and the largest \|Δ log-prob\|, each held to the reference's own bf16-vs-fp32 spread times `threshold_factor`; the verdict quotes the worst position (`checks/logit_agreement.py`) |

All of them run with **`modeling_v2=require`** (the gates export
`TRTLLM_MODELING_V2`, and the harness hands it to the LLM API). Under `auto` a
configuration that misses a target's routing criteria silently gets the
built-in implementation, and a number measured that way reads as
modeling_v2's — the single most expensive mistake available in this package.

**Every gate attests the engine it ran.** A gate's argv and handed-over
KV-cache keys are what it *asks*; the LLM API and the executor's worker can
change them (a hybrid model with no snapshot policy runs without block reuse,
and its gates still generate the expected text). Once the engine is up, the harness prints one
`MV2FLOW_ENGINE` record (`checks/harness/engine_facts.py`): `modeling_v2`,
block reuse, the Mamba snapshot policy, the KV and SSM cache dtypes (as asked
and as resolved), the CUDA-graph batch list and padding, the speculator and
its depth, `max_num_tokens`, `max_batch_size`, chunked prefill, TP, EP and
attention DP.

- **Where each value comes from.** Every rank's own worker: the gates put
  `checks/harness/rankhook` first on `PYTHONPATH` and set
  `MV2FLOW_RANK_FACTS=1`, and its `sitecustomize` has each executor worker, in
  whatever process it runs, print one `MV2FLOW_RANK_ENGINE` record as its
  engine is set up (the same reading as below, and its rank). A field every
  reporting rank carries is held to the declaration on each rank, and a
  difference names the ranks. Without those records: rank 0's worker when it
  sets its engine up in the harness process (`TLLM_WORKER_USE_SINGLE_PROCESS=1`
  at one rank), otherwise `llm.args`, without block reuse and the overlap
  scheduler, which only the worker decides; `llm.args` is the request, and it
  agrees with any declaration. Block reuse then comes from the log, when a
  rank logged turning it off, or from what the engine reused (below), and the
  KV pool from rank 0's executor through `LLM.get_kv_cache_capacity`.
- **What a check is about must be observed.** `checks/observables.py`
  registers, per feature, the evidence that it took effect and the oracle it
  is judged by: a cache dtype as each rank's worker resolved it; chunked
  prefill as a request whose prompt, past what the cache resumed, exceeds
  `max_num_tokens` (the boot of a chunked variant sends the reference's long
  prompt, or a chunk probe of twice `max_num_tokens`); block reuse as the
  reuse probe; a speculator as acceptance per prompt. A configuration
  variant is about the fields its overlay sets, and its boot fails when one
  of them is not observed: read only from `llm.args`, or with a rank that
  printed no record. A variant that changes only lossless settings (a chunk
  size, block reuse, the capture list) is held to the HF reference's greedy
  tokens as its unit is.
- **The comparison.** Boot, both acceptance legs, both envelope legs and every
  witness leg the quickstart shim runs compare each field with what the
  unit's resolved runtime declares (`gates.engine_declaration`: the gate argv,
  the options file's snapshot policy and cache dtypes). Any difference fails
  the gate with the fields named. A field the record lacks is reported as
  `NOT ATTESTED`, never as agreeing. The accuracy gate runs `trtllm-eval`
  under the eval shim (`checks/harness/eval_shim.py`), which prints the same
  record; of its configuration block reuse and the cache dtypes are compared,
  with the options file's `enable_block_reuse` and `kv_cache_config` dtypes. Leg B of the equivalence witness builds its
  engines outside the shim, so their configuration is not attested. A
  witness leg the shim runs that asks for block reuse and reused nothing,
  although it had a prefix to reuse, fails the witness as it fails boot.
- **What the engine reused.** Every response carries `cached_tokens`, the
  prompt tokens rank 0's executor resumed from the KV cache, at every world
  size. After the workload the harness prints one `MV2FLOW_REUSE` record
  (`checks/harness/engine_reuse.py`): the requests that resumed, the tokens
  and positions they resumed at, and how many could have (their prompt shared
  more than the engine's shortest resumable prefix with an earlier one: the
  first snapshot point of its policy, else one KV block). For a hybrid model
  each resume restored a recurrent-state snapshot, and the record counts them
  as `snapshot_restores`. The boot, acceptance and envelope drivers then send
  the reuse probe (the envelope's in each leg, so also once the engine has
  served at its limits): one prompt two KV blocks past that prefix, twice, the
  second after the first, on attention-DP rank 0. The probe attests block reuse where no
  worker was read, so every member attests all its fields. A unit that asks
  for reuse fails a gate where the engine reused nothing although it could
  have (`gates.reuse_shortfall`): the probe's second prompt computed from token
  0, or more requests than the engine runs at once could have resumed and
  none did.
- **Pools and memory.** Each verdict carries the recurrent slots every rank
  logged (`bound <manager> #N: <slots> recurrent slots`), rank 0's KV-cache
  tokens and the free memory of each device once the engine was up (NVML).
  The journal line shows them as `[engine 12/12 fields as declared; recurrent
  slots 4096 x4; KV 262144 tokens; reuse probe resumed at 256 of 320
  (snapshot restored); min free 23.97 of 276.62 GiB]`, so pool sizes compare
  across runs, and an accuracy line as `[engine 1/1 fields as declared; reuse
  1318/1319 requests, 1012224 tokens (recurrent state restored at 768 x1318);
  ...]`.
- **Margin.** A device under the larger of `runtime.min_free_memory_gib`
  (default 2) and `runtime.min_free_memory_fraction` of the device (default
  0.02) still passes, with `[WARNING: LOW MARGIN: ...]` in its summary. Both
  keys work per unit in the runtime block.
- **Runtime facts.** Each gate that passes adds its engines to
  `runtime_facts.yaml` (`run/runtime_facts.py`), the latest per gate and
  engine leg for its unit and member: the CUDA-graph batches captured and
  their padding, the recurrent slots per rank, the KV pool, the free memory,
  the speculation depth, TP, EP, attention DP, `max_batch_size` and
  `max_num_tokens`. A member's facts span its engines on the target as it is
  (or the latest, marked `stale`); the envelope spans every member. A gate
  whose own numbers moved since its last pass (not free memory or the KV
  pool, which follow the device's other tenants) is a `runtime facts
  changed` journal event and a line in the file's `changes`. The planner's
  request carries the key numbers and the latest changes; every smith's and
  worker's prompt carries the key numbers and the path, and the smith is
  told to certify against them.
- **Serving geometry.** Before any kernel is certified, the geometry probe
  (`run/geometry.py`, harness `checks/harness/geometry_probe.py`) boots the
  engine once per width under the run's runtime block: the narrowest member,
  the widest one that is not attention-DP, the widest attention-DP member,
  and the narrowest again under the first unit that turns speculation on. It
  records under `geometry` in `runtime_facts.yaml` what each serves: the
  capture list and padding, the KV pool and its page view (tokens per block,
  pages, pages per sequence), the limits, the snapshot policy, and the
  context chunks prompts of each probe length ran in (read from the
  iteration statistics; the lengths cross the token budget, the snapshot
  interval and offsets, and the witness's prompt lengths), plus each
  speculative unit's declared draft depth. It runs the stock path; for an
  architecture the stock path does not know it boots
  `environment.substrate_checkpoint` and keeps only what the configuration
  decides (capture list, limits, page size, widths). Every prompt's runtime
  facts block carries it with the values a verify row's `cells` state; a
  gate's first pass that serves different numbers is a `changes` line. It
  never stops the run, a resumed run whose file holds a probe of the same
  inputs does not repeat it, and a smith dispatch waits for a running probe
  at most 15 minutes. The first time the first unit boots on the target at a
  width, that width's legs are probed again on the target (`require`),
  beside the other legs; widths whose core boots meanwhile go in the next
  probe together. The values the target serves that no earlier probe had
  are one planner event naming the entries certified so far, so
  re-certification is planned once, as a batch
  (`geometry.recertification_note`).
- **Covered certifications.** A `verify` dispatch that states its required
  cells (the queue row's `cells`, or a ```` ```cells ```` YAML list in its
  task's goal; each cell a mapping of the entry's own spec keys) is closed
  `LANDED` without a smith turn when the entry and its test are at the
  checkout's `HEAD` and every required cell is a cell of its `CELLS` on the
  run's architecture in its `ARCHS` (`run/covered.py`, read statically by
  `onboard/coverage.py`). Its decision file and receipt name the commit and
  the covering cells. A missing cell, an uncertified architecture, a value
  the source does not state readably, or a task branch with commits of its
  own dispatches the smith as before.

The **acceptance** gate exists because boot and accuracy are blind to
speculative decoding: rejection sampling holds the emitted distribution to the
target model's, so a *miscomputed* draft path produces correct text more
slowly. Only speed moves.

Under `parallel: unified` — the default — the gates run **once per member**,
where the members are the concrete configurations the one codebase serves
(`members:` in `task.yaml`, default `[tp<W>, tep<W>, dep<W>]`), cheapest first.
A unified target gated at a single topology leaves the expert-parallel and
attention-DP branches unexecuted while its Coverage claims them. Every verdict
names its member (`accuracy[gsm8k]@dep4`), and the contract holds only when
all of them pass. `--dry-run` prints the member set and the resulting gate count
before anything is spent.

### Gate workloads, floors and findings

Every gate's workload is declared in `task.yaml` under `accuracy`, never sized
from the run it judges. Each key also works inside one unit's entry, for that
unit only.

- **Accuracy floor** (`accuracy.floor`).
  - `rule: statistical` (the default) puts the floor `k` standard errors
    below the anchor; `k` defaults to 2, and the SE comes from the
    benchmark's sample count (`samples` when the gate cannot read it).
  - `rule: tolerance` keeps the fixed `tolerance` width instead.
  - If *every* member of a unit scores more than 1 SE below the anchor, the
    unit passes but the run records a **finding**. Delivery refuses it
    (`delivery-failed:findings`) until `floor.acknowledge.<unit>` gives the
    reason.
- **Acceptance** (`accuracy.acceptance`).
  - The workload is `num_prompts` (default 32) questions from the accuracy
    dataset (`dataset`, read offline), formatted with `prompt_template`. A
    run may instead declare `prompts` or `prompts_file`.
  - Fewer than `min_prompts` prompts fail as "workload too small". A workload
    that reaches fewer than `min_draft_iterations` draft iterations fails as
    "too short".
  - Each unit's run is compared prompt by prompt with the previous verdict
    (`regression: {test, alpha, min_effect}`).
  - A stock leg that cannot load the checkpoint is recorded per checkpoint
    and GPU architecture (`gate_results.jsonl`), and skipped from then on
    where a floor or a shallower depth stands in.
  - Where the stock path cannot load the checkpoint, a draft depth above 1
    of a speculator whose depth is a setting (MTP, Eagle3; not a DSpark
    drafter, whose block is the checkpoint's) is compared with the same
    target one depth shallower, on the same
    prompts, in the same job: the deeper must accept significantly more
    (one-sided paired test at the regression `alpha`). A draft position that
    is never accepted adds nothing, so a first verdict is checked rather than
    taken as the baseline.
  - Per unit, set `min_absolute` with a `floor_source`, or set
    `measured_reference`; at depth 1 without a stock path it is the only
    comparison. A floor with no measured source is warned about.
- **Serve** (`accuracy.serve`).
  - On by default wherever the spec bounds the sequence length the server
    sizes its cache for (`accuracy.max_seq_len`, or `max_seq_len` in the
    run-wide options file); off where it does not, since the server would
    size its cache for the checkpoint's own maximum. The loader writes the
    value it chose into the workspace spec. `enabled: false` turns it off; a
    run resumed with a spec that does not mention the block owes it from then
    on when it can be derived.
  - `prompt` (default "The capital of France is") and `max_tokens` (default
    32) are the completion request; `timeout` (default 1800) is how long the
    server may take to answer `/health`.
- **Logits** (`accuracy.logits`).
  - On wherever `reference` names a reference file. A spec that names none,
    and does not say `enabled: false`, gets `logits.reference.json.gz` beside
    the spec when that file is there. With it the accuracy benchmark is
    sampled (`benchmark_members`, below) and the boot compares the
    reference's greedy tokens. The file is made once, in the anchors phase, on
    the HF implementation:
    `python <agent-flow>/agent_flow/workflows/trtllm_modeling_v2/onboard/logit_reference.py
    --model <checkpoint> --out logits.reference.json.gz [--trust-remote-code]
    [--image <repo-relative png> --image-root <tree>] [--prompts prompts.yaml]`.
    It loads the checkpoint twice (bf16, then fp32 for the spread), so give it
    room for both, one after the other (`--device-map auto` spreads them).
  - The default prompts are three short raw-text prompts and a long prompt of
    8704 tokens, one context chunk past the default `max_num_tokens`; make it
    longer than the unit's own `max_num_tokens` (`--long-tokens`) so it
    crosses a chunk, and the gate warns when nothing does. The long prompt's
    context logits are held in device memory (8704 × vocabulary × 4 bytes),
    which the KV-cache fraction must leave room for.
  - `prompts` picks reference prompt ids; by default every one the unit can
    take (an image prompt only to a multimodal unit; a prompt past
    `max_seq_len`, or past `max_num_tokens` without chunked prefill, is left
    out with a warning). `threshold_factor` (default 3) scales every limit;
    `timeout` (default 1800) bounds the job.
  - Lossy configurations (a quantized KV cache, a narrower state dtype) move
    the logits by design, so the configuration variants owe no logits check,
    their boots compare no greedy tokens unless their own
    `accuracy.boot.<variant>.greedy` turns it on, and their benchmark runs at
    every member they name.
- **Where the benchmark runs** (`accuracy.benchmark_members`).
  - `sampled` (default): with the logit gate owed, a unit's benchmark runs at
    its narrowest member, its widest (the cheapest to start among equals) and
    every member whose served engine the logit gate does not run whole: an
    attention-DP member (the gate sends one prompt at a time, which one rank
    serves), and every member of a unit with a speculator when its boot
    compares no greedy tokens (the gate builds the engine without it). The
    logit gate at every member proves the rest equivalent to the reference.
    Without the logit gate the benchmark runs at every member.
  - `every`: at every member regardless.
- **Boot greedy tokens** (`accuracy.boot.greedy`, run-wide or a unit's own).
  - On wherever the logit reference is named, logit gate on or off; `enabled:
    false` compares none. `prompts` (default 4) of the reference's text
    prompts that fit the engine, `tokens` (default 24, at most the
    reference's continuation) each. Every boot request then generates its
    whole budget past any end token. A level holding only `greedy` declares no
    workload, so the workspace `boot.yaml` still applies.
  - An image prompt needs a target whose input processor keeps the image
    placeholders' vocabulary ids: TensorRT-LLM looks up each prompt token's
    log-prob, and an id past the vocabulary fails the request. Leave image
    prompts out with `prompts` for such a target.
- **Boot** (`accuracy.boot`, else the workspace `boot.yaml`).
  - `prompts` may carry their own `keywords` and `images`, plus `max_tokens`.
  - `multimodal.prompts` gives the image prompts. Without one, the
    repository's `merlion.png` is sent; it is a git-lfs file, so pull it in
    the tree under test.
- **Witness workloads** (`accuracy.witness.<feature>`).
  - Token-comparing witnesses declare `prompts`/`prompts_file`, `max_tokens`
    and `min_generated_tokens` (default 32). A continuation too short to
    compare is refused, not passed.
  - Both legs print token ids (`[i] Witness token ids: [...]` under
    `TRTLLM_WITNESS_FULL_OUTPUT=1`). Text that agrees without them is
    refused, not passed.
  - The equivalence witness's `prompt_tokens` prompts are prose
    (`checks/witness_text.py`): that many words, so at least that many
    tokens, and about 16% more under the one tokenizer measured, so leave
    `max_seq_len` room. A counting pattern is continued from the attention
    pages alone and cannot show a hop that drops the recurrent state.
  - **The equivalence self-test.** Before a pass, leg B runs once more on the
    longest prompt with `MV2FLOW_WITNESS_ABLATE=state`, which destroys the
    state the generation engine received. That continuation must differ from
    leg A's. If it does not, the verdict is `INSENSITIVE, not a pass`.
  - **Attention DP.** At a `dep` member, leg B is handed
    `MV2FLOW_WITNESS_RANK_PAIRS`, one (context rank, generation rank) pair
    per prompt covering every pair (`0:0,1:1,0:1,1:0` at dep2). The prompts
    repeat on both legs until there is one per pair. Unpinned, every hop
    goes from rank 0 to rank 0.
  - **Leg B.** Without a `contrast_command` the witness runs the flow's own
    (`checks/harness/disagg_leg_b.py`), so no run writes one. It splits the
    node's devices into a context and a generation engine of the member's
    width (and refuses when there are fewer than twice the member's ranks),
    builds both from leg A's quickstart argv through the tree's own
    `quickstart_advanced.py`, with the block's `cache_transceiver_config`
    (default `{backend: DEFAULT, transceiver_runtime: auto}`) and no overlap
    scheduler on the context engine, and sends each prompt context-only, then
    generation-only with the `DisaggregatedParams` it got back. It honours
    every variable below. Its ablation reaches the generation engine's
    processes through a `sitecustomize` first on their `PYTHONPATH` (which
    runs the next one, such as a session's resolve hook), and acts where the
    executor takes up a request whose transfer completed
    (`PyExecutor._prepare_disagg_gen_transmission_complete`). An engine whose
    caches it cannot reach (no recurrent layers with conv and SSM state, no
    KV manager that names a request's pages) is refused when it is built:
    `Witness ablation refused: ...`.
  - **What a `contrast_command` honours**, beyond reading leg A's workload
    from the `MV2FLOW_WITNESS_*` variables (see `checks/witness.py`), when a
    run names its own launcher (then `contrast_env` is required too):
    - `MV2FLOW_WITNESS_RANK_PAIRS`: pin prompt `i`'s context-only and
      generation-only requests to the `i`-th pair
      (`SchedulingParams(attention_dp_rank=..., attention_dp_relax=False)`),
      and print `[i] Witness ranks: ctx=<c> gen=<g>`. A hop without that line
      fails the witness.
    - `MV2FLOW_WITNESS_ABLATE=state`: on the generation engine, after each
      generation-only request's transfer completes and before its first
      decode step, zero the recurrent state (conv window and SSM state) of
      every layer that has one, else the request's KV pages. Print
      `Witness ablated: <what>` for each such request, from any process.
      Without that line the self-test cannot run and the witness fails.
- **Pinned tuning.** The token-comparing witnesses run each leg as its own
  process sharing the AutoTuner's tactics and Triton's autotuning
  (`checks/gates.py`, `pin_tuning`), emptied at the start of each gate run, so two legs of
  one configuration do not differ in the last bit.

Per-prompt baselines and findings are kept in `<workspace>/gate_results.jsonl`.

A gate whose compute session was lost under it measured nothing. It is not
recorded and costs no assembler turn: the orchestrator waits up to
`environment.session_wait` (default 1 h) for a session to heart-beat again and
re-runs it, at most three times in a row. Past either bound the run closes
`session-lost:<stage>`, and re-running the same command resumes at that gate.

**Running one gate by hand.** `trtllm-modeling-v2 gate
{boot,accuracy,acceptance,envelope,witness,serve,logits} --task task.yaml
[--unit U] [--member M] [--commit C | --tree T] [--results F] [--workdir W]`
runs a single gate or witness exactly as the orchestrator does: on a snapshot
of the commit the checkout is at, or of `--commit`, kept with the run's own
snapshots when `--task` is the workspace's `task.yaml` (`--snapshots` puts
them elsewhere). With `--tree` it measures that tree as it is (a worktree
with uncommitted work, another checkout), which is how a historical defect is
replayed on the trees before and after its fix; its verdict names that tree
(`git describe`, `-dirty` when the target's path is uncommitted), a witness's
included. Assemblers are pointed at the same command.

**Not built yet:**
- a performance gate: one prefill chunk plus one decode step against the
  previous tree, failing on more than 5% (the engine driver's timing mode
  exists);
- a concurrent, batch > 1 leg beside the batch-1 witness legs.

### Tests after a shared change

A change to code other models run is followed by a cross-model regression
review, which runs those models' unit tests. It runs only the ones the
change can affect, and reuses every other result from the run's test cache,
`<workspace>/test_cache/`:

```
trtllm-modeling-v2 tests affected --workspace W --head <commit> --paths <test dirs>
<tree's python> checks/testcache_driver.py run --cache W/test_cache --repo <checkout at head> \
    --selection <the file it wrote> -- -p no:cacheprovider -rA
trtllm-modeling-v2 tests compare --workspace W --head <commit> --paths <test dirs>
```

The driver runs one pytest process per test file and records, for each test,
its outcome and everything the process ran on before it ended, each input
with the test it was first seen in (`checks/testcache_driver.py`): the
tree's modules it loaded, the functions that started (`sys.monitoring`), the
files it read, the directories it listed, the paths whose existence it
asked, the module names an import searched for and did not find in the
tree, and the tree's shared libraries. A recorded result holds at another
commit when the test ran there, or when none of that differs: a changed
module counts only through a function body the process ran, unless the
change is outside function bodies (module-level code, a class body, a
signature, a decorator) or in a function a decorator compiles or traces; a
native source change counts for every test of a process that loaded the
tree's library. A test from the first process it starts that can run Python
or build code, or from the first frame `torch.compile` compiled, is reused
only at its own commit, and results hold only in the environment they were
recorded in (interpreter, packages, GPUs, the settings variables).
Cumulative inputs are what make a memoized result safe: a test that only
reads back what an earlier one computed still depends on what computed it.

`tests affected` writes the tests no result holds for to a selection file (a
whole file where what its collection ran changed, so the test ids may have)
and prints the command. `tests compare` then reads the head's results
against the base's (`--base`, else the commit the last regression review
`cleared`, else the parent of the run's first integrated commit):
`REGRESSED` names each test that passed at the base and fails now, or is new
and fails; `INCOMPLETE` lists what still has to run, at the head or, for a
failure with no base result, at the base; `CLEAN` lets the review approve.
The base tree is run only for a test that fails at the head and has no
result that holds at the base.

A smith's certifying run of its entry's test goes through the same cache, so
the result is recorded with its commit. A certification whose cells the
entry already lists is closed without a turn only when the cache holds a
pass of that test file that still holds at the head (`run/covered.py`), and
delivery requires the delivered tests to pass at the head (below). A process
counts as running code by its program (`checks/testcache_driver.starts_code`):
`file -b /usr/bin/python3.12`, which `platform.architecture()` starts, names
an interpreter and runs none.

## The serving envelope

The runtime block's limits — `max_batch_size`, `max_num_tokens`,
`max_seq_len` and the CUDA-graph capture batches — are the serving envelope:
what the target is claimed to serve. The roles judge against it, never
against the shapes one gate happens to launch:

- the plan roles mark a vocabulary row `CERT`, not `OK`, when its entry's
  contract stops at the gate workloads;
- each vocabulary step's `call_sites` give the rows per request at each
  place the target calls the entry, and the smith multiplies them by every
  captured batch size to get the surface it certifies;
- the reviewer checks every capacity constant in the target against the
  envelope (a constant is derived from an engine argument or carries a
  `# capacity:` binding on its line).

The capture batches are one list per unit (`gates.resolve_capture_list`):
the options file's `cuda_graph_config` when it sets one, else the LLM API's
`CudaGraphConfig` rule over the runtime block's flags. The accuracy gate's
`trtllm-eval` reads it from the options file; boot, acceptance and envelope
get it as `--cuda_graph_padding_enabled` / `--cuda_graph_batch_sizes`, as
they get the draft depth as `--spec_decode_max_draft_len`. Preflight
(`check_capture_lists`) refuses a unit whose runtime flags would still make
the two launch paths capture different lists, naming both. The survey turn
is given each unit's list beside the LLM API default and tune's lists, and
the plan's envelope section states them.

The KV-cache configuration is one per unit too. The quickstart has flags for
only some `KvCacheConfig` keys; the rest of the options file's
`kv_cache_config` (`gates.resolve_kv_cache_config`), `mamba_state_config`
above all, goes to the boot and acceptance engine driver and to every witness
leg as `MV2FLOW_KV_CACHE_CONFIG`, and the harness merges it into the
`KvCacheConfig` the quickstart builds; a key a quickstart flag sets keeps the
flag's value. Without a Mamba state snapshot policy the engine turns block
reuse off for a hybrid model, so a boot, acceptance or accuracy gate whose log
says it did fails, naming the setting (`gates.reuse_disabled_detail`), and
preflight notes each unit that asks for reuse on a hybrid checkpoint without
one (`check_kv_cache_configs`).

## Delivery

Once the planned run's contract holds, delivery verifies, in order:

1. **Findings.** An unacknowledged accuracy finding closes
   `delivery-failed:findings`.
2. **Disk** (`onboard/delivery.py`). The six trees a finished onboarding
   touches: the catalog entries the target imports (shared ones under
   `catalog/`, its own under `ops/`), their GPU tests, the target, its routing
   entry, the accuracy anchor and the test-list registration; `ops:promoted`
   (`promotion.check_promoted`): no local variant left in the target or
   imported by it, but those `target.keep_local` names. Then
   `git:target-committed` and `git:commit-hygiene`, and `tests`: the
   target's own tests and those of the catalog entries it is delivered with,
   by the newest result in the run's test cache that holds at the head
   (`testcache.head_results`): none failing, none still to run. A failure
   closes `delivery-failed`.
3. **The contract** (`run/contract.py`) on the target as it is now. Each
   verdict records the fingerprint of the tree its check measured (see
   [The pipeline](#the-pipeline)); one recorded against another state counts
   as missing, so a target that changed after it was gated (its own files, a
   catalog entry it imports, the native build) is not delivered on old
   evidence. A failure closes `delivery-failed:contract`. Its journal
   section also says where correctness stands delivered
   (`correctness.jsonl`).

Records are never overwritten. Decisions are
`decisions/<unit>-<role>-<turn>.yaml`, and a reused key gets `.2`, `.3` and so
on; an external driver must write the decision path its request names,
verbatim. Journal ordinals are claimed atomically.

The catalog is its entry files. A smith commits one entry file and its test
(a shared entry's local variant, until the promote), and nothing else; there
is no index to regenerate. The survey lists the shared
entries from `catalog/<category>/*.py` themselves, and a target's own ops from
its `ops/`.

## The roles

| role | runs | one turn does |
|---|---|---|
| `planner` | at the start and on every event | writes the next task graph; `GRAPH_UPDATED`, `DONE`, `ESCALATE` or `CEILING` |
| `plan_drafter` | a `survey` task | the checkpoint→catalog mapping and the plan |
| `plan_reviewer` | a `review` task with `ref: plan` | a paper review of both |
| `smith` | a `kernel` task | exactly one catalog entry |
| `assembler` | an `assemble`, `feature`, `integrate` or `investigate` task | the core, one feature, a merge or a diagnosis |
| `reviewer` | a `review` task | reads one unit's diff against the gate evidence |
| `tuner` | inside a `tune` task | one tune hypothesis, or a roadmap |
| `improver` | when the run closes, on what cost it time (a failed task, a re-certification, a ready task waiting, a no-op planner revision, a reported flow issue, a void verdict); beside the run too in one that measures the workflow (`knowledge: measurement`), on idle agent slots | lessons, flow issues and patch drafts; see [Lessons](#lessons-and-the-improver) |

Every role is stateless: tasks run concurrently, and what a turn needs from
earlier ones is on disk. Every role runs on the agent that drives the run, at
its model and reasoning effort, unless `task.yaml` routes the role elsewhere
(below).

### Choosing each role's backend and model

An optional `agents` block in `task.yaml` routes a role away from the driving
agent: the backend (`claude-code` or `codex`), model, reasoning effort and
extra MCP servers, for every role (`defaults`) or for one (`roles.<name>`).
Use it for a specific need -- a plan reviewer on another provider for a
cross-model review, say -- not by default. It is the shared per-role routing of
`agent_flow.agent_runtime`, the same block `perf-analyze` and `perf-optimize`
read, with the role list and defaults in `agents/roles.py`:

```yaml
agents:
  roles:
    plan_reviewer: {backend: codex, model: gpt-6-astra, reasoning_effort: high}
```

- A role on Codex gets no Stop hook, because Codex refuses Claude SDK hooks;
  its decision file is validated when it is read, like every role's.
- Under `--agent-transport external` each `dispatch/<id>.request.json` carries
  the role's configured backend and model as `agent`, and running the turn on
  a matching model is the driver's job.
- The spec is re-read on every resume (`_refresh_workspace_spec`), so a
  changed `agents` block applies on the next resume.

Above `AGENT_FLOW_SYSTEM_PROMPT_FILE_THRESHOLD` (64 KiB) the Claude backend
writes a system prompt to `<workspace>/.agent-flow/system-prompt-<sha>.md` and
passes a pointer, because Linux caps one argv string at 128 KiB. The large
per-run blocks (survey, runtime areas, kernel sources) travel in the turn
prompt, which is not on argv.

## Workspace layout

```
<workspace>/
├── task.yaml             the resolved spec, defaults filled in
├── prompts/<role>.md     each role's composed system prompt, snapshotted at every launch
├── contract.yaml         every check the run owes (preflight)
├── coverage.yaml         what the checks prove of the request, feature x member (preflight)
├── contract_status.jsonl every verdict recorded against it, with the commit it measured and what each command imported
├── graph.yaml            the task graph the orchestrator adopted, with every revision's reason
├── graph.proposed.yaml   the planner's last proposal, before validation
├── planner_events.json   what happened since the planner's last turn, and each task's infrastructure retries
├── correctness.jsonl     each commit correctness was delivered at (every required check held there), and the bring-up's (tier `bringup`)
├── tasks/<id>.json       each task's envelope: goal, touches, satisfies, priority, its dependencies' receipts
├── tasks/<id>.receipt.yaml  a finished task's receipt, handed to the tasks that depend on it
├── runtime_facts.yaml    what the engines of passed gates ran, per unit and member; the probe's serving geometry
├── gates/geometry/       the geometry probe's leg specs
├── plan.md               a `survey` task's plan
├── vocabulary.yaml       the typed step→catalog mapping and dispatch queue
├── surface.md            the surface survey, unabridged (preflight)
├── host_facts.md         host facts every dispatch is pointed at; roles append below the marker
├── worktrees/<task-id>/  each tree-changing task's own branch of the checkout (removed once integrated)
├── snapshots/<commit>/   the read-only snapshot of a commit the checks measure (removed once no check needs it)
├── progress.yaml         every agent's typed decision
├── decisions/<unit>-<role>-<turn>.yaml   typed decisions, one file per turn
├── journal.md            the run's timeline, written as events happen
├── reports/NN-<role>-<what>.md   each dispatch's prompt and result
├── gates/<unit>/         gate working files (on the shared filesystem, read by the job)
├── gates/<unit>/bundles/ each gate bundle's spec, `<name>.bundle.json`
├── gates/agent-probe/    preflight's agent probe: each agent's turn and file under <n>/, the last pass in pass.json
├── turns/<id>.*          each in-process turn: request.json, system.md, prompt.md, activity.log, result.md, done
├── witness/<unit>/       witness working files
├── gate_results.jsonl    per-prompt gate baselines and open findings
├── test_cache/sessions/  every recorded pytest process: each test's result and what it ran on (`tests affected`)
├── tune/                 the tune campaign: worktrees, measurements, perf_ledger.jsonl, summary.md
├── lessons.proposed.yaml the improver's last proposal, before validation
├── flow_issues.yaml      the workflow defects the improver filed: the operator's fix list
├── flow_issues.reported.jsonl  flow issues the driver and the workers reported, one JSON object per line
├── improve/wakes.json    what woke the improver and no turn has read yet, and what woke it before
├── improve/patches/<lesson id>.diff  the improver's patch drafts, each a fix and its test; never applied
├── boot.yaml             the boot gate's prompts and keywords (optional)
├── bin/commit.sh         the locking, self-verifying commit helper
├── .commit.lock          its lock file
├── .claude/skills/       this run's copy of the packaged skills
├── .agent-flow/          system prompts too long for argv, when there are any
├── dispatch/             the orchestrator's heartbeat and exit note; under --agent-transport external also the turns, each with what the orchestrator did with it (.read, .adopted, .consumed)
├── TURN_INSTRUCTIONS.md  the driver's standing instructions, under --agent-transport external
├── inflight.json         what the orchestrator holds that a stop would cut short, and how it last stopped
├── restarts.jsonl        one line per orchestrator start: what the stop before it cost
├── reload.request.json   a reload the driver asked for and the orchestrator has not answered yet
├── reload.json           the latest reload: draining, re-executed, reloaded, cancelled or failed
└── .trtllm_modeling_v2_state.json
```

The journal and the reports stay here rather than in the repository: they are
evidence *about a run*, and a product tree is the wrong home for it.

`vocabulary.yaml` is the one artifact with a schema, because the orchestrator
walks it. `onboard/vocabulary.py` validates it and rejects a malformed one:
`OK`/`CERT` rows quote the cell, `note` line or `is_valid` assertion that
decides coverage, a queue `target` is `<category>/<name>` (shared catalog) or
`ops/<category>/<name>` (the target's own `ops/`) and has no `__` in its name,
a `site` is a shared row's and no two rows land on one variant, `GAP` rows
name a candidate, `verify` dispatches name the surface, the two chapters do
not overlap, and a `ceiling` verdict carries its evidence. A `survey` task whose
`vocabulary.yaml` does not validate has failed.

A `GAP` row names its rung on the sourcing ladder, cheapest first:

| tier | source | the row names |
|---|---|---|
| 1 | a `torch.ops.trtllm.*` op | the op |
| 2 | an in-tree callable under `tensorrt_llm/_torch/modules/**` | the callable |
| 3 | FlashInfer's public API, for a kernel no trtllm op wraps | `flashinfer.<module>.<name>` |
| 4 | native torch, composed by this run | what tiers 1-3 offered (`tried`) |
| 5 | a Triton kernel, written by this run | what tiers 1-3 offered (`tried`) |

The op inventory sees tier 1 only. The surface survey also reads the installed
FlashInfer's public API off disk and records which callables the engine
already references; `surface.md` lists them. A ceiling accounts for all five
tiers.

Each queue row becomes a `kernel` task when the planner decides; the row's
`unit` and `depends_on` tell the planner which assembly needs it first. A smith
that answers `MACRO` names its constituents, and the planner adds them as
`kernel` tasks of their own (`ref: <mode>:<entry>`).

## Package layout

```
trtllm_modeling_v2/
├── cli.py            the command line and its subcommands
├── spec/             what the user asks: task_schema, schema_keys, intake/ (Step 1)
├── run/              the run: contract, graph, orchestrator, workflow, state, journal,
│                     progress, preflight, receipts, runtime_facts, geometry (the
│                     probe), covered (a certification the tree holds), stages/ (one
│                     mixin per concern)
├── onboard/          what onboarding knows: survey, vocabulary, runtime_map, delivery,
│                     promotion (local variants and their promote), embargo,
│                     logit_reference (the logit gate's HF reference, made in the
│                     anchors phase), coverage (an entry's CELLS read statically),
│                     runtime_index.yaml, kernel_sources.yaml
├── checks/           the contract's checks: gates, gate_records, witness, envelope_boot,
│                     serve_gate (and serve_driver, the job it launches), logit_agreement
│                     (and logit_driver), bundle (one engine for a member's checks), stats,
│                     gate_cli, harness/ (the engine driver the gates launch, and the
│                     bundle driver), testcache (and testcache_driver, the recorded pytest run)
├── tune/             the tune campaign: stage, campaign, keep_gate, measure, records, inert, serving_config
├── agents/           roles, prompts/, skills/, lessons (the improver's store)
├── infra/            not modeling: environment, transport, execution/, driver/,
│                     substrate/ (the spool scheduler), scripts/, committer, vcs
└── examples/         worked examples for operators; never handed to a run
```

## Flags

| flag | default | |
|---|---|---|
| `--task` | `--task` or `--checkpoint` | the spec. Validated before any agent is constructed |
| `--checkpoint` | `--task` or `--checkpoint` | a HuggingFace checkpoint: derive the task from its `config.json`, `--site` and `--goal` into `<workspace>/task.intake.yaml` and run it; its intake checks run in phase 0. Re-running the same command resumes; `--clean` derives again |
| `--site` | none | the cluster's `site.yaml`: required with `--checkpoint`; with an intake `task.yaml` (see INTAKE.md) the two compile into `<workspace>/task.resolved.yaml` |
| `--goal` | none | with `--checkpoint`: features, members and workload in one line, e.g. `"mtp disagg@tp1/tep4 tp1 tep4 isl=8192 osl=1024 tune"`; `-<feature>` leaves out one the checkpoint carries (write `--goal=-mtp ...` when the line starts with `-`) |
| `--confirm` [BY] | not confirmed | with `--checkpoint`: record that BY (default `$USER`) confirms the derived task. The run does not wait for it; an edit after the run started is refused until confirmed |
| `--workspace` | `workspace/trtllm-modeling-v2` | |
| `--dry-run` | off | preflight, then stop. Leaves the checkpoint at `preflight`, so the real run re-runs it |
| `--max-parallel-tasks` | `8` | how many agents work at once, shared by priority (`infra/execution/agent_gate.py`): the planner owns one slot and goes first, then tasks on the critical path, other required tasks, the critic and improver, and last a tune campaign's turns, which use only the slots the open feature work leaves idle; a task holds a slot only while its agent works, not while it waits on a soft dependency or runs its checks, and checks (8 at a time) take no agent. Under `--agent-transport external`, `driver wait` keeps at most 8 turns in flight |
| `--revision-cap` | `6` | graph revisions without a contract check turning green before the run closes `planner-stalled` |
| `--no-task-worktrees` | off | tasks edit the checkout directly instead of each in its own worktree; see [The pipeline](#the-pipeline) |
| `--no-graph-critic` | off | skip the advisory read of the first task graph and of reshaping revisions |
| `--event-window` SECONDS | `60` | while tasks, checks or a planner turn are in flight, how long a task that finished with its checks green waits to reach the planner with the events after it; a failure, a worker's report, a critique, a failed re-check, or nothing left to run or start wakes the planner at once |
| `--lessons-dir` | `$MV2FLOW_LESSONS_DIR`, else `~/.trtllm_modeling_v2/lessons` | where the improver keeps lessons across runs; see [Lessons](#lessons-and-the-improver) |
| `--no-lessons` | off | neither show lessons nor learn new ones: an evaluation run that starts from what the workflow ships |
| `--wait-for-spool` MINUTES | 0 | with a `session_spool` environment, wait up to MINUTES for the spool to appear (its session is still importing the image) before validating the task |
| `--skip-substrate-gate` | off | skip the stock-path cold start in preflight. An escape hatch, not a speed-up — see the flag's own help |
| `--agent-transport` | `inprocess` | how a role turn is delivered. `inprocess` runs it here — see [Running the turns in process](#running-the-turns-in-process); `external` writes it to `<workspace>/dispatch/` and waits — see below |
| `--agent-turn-timeout` | per role | seconds one role turn may take, in process or dispatched, for every role. Without it each role has its own bound: assembler 16 h, tuner 8 h, smith 6 h, plan drafter and reviewer 4 h, plan reviewer 3 h, planner 2 h |
| `--stage` | — | `tune`: tune a target onboarded elsewhere, or re-enter a closed run to tune it further. Same as the `tune` subcommand |
| `--clean` | off | wipe the checkpoint and managed files |

Subcommands:

| command | does |
|---|---|
| `trtllm-modeling-v2 tune --task task.yaml --workspace W` | tune a target on its own — see [Tuning](#tuning) |
| `trtllm-modeling-v2 gate <name> --task task.yaml [--unit U] [--member M] [--commit C \| --tree T]` | one gate or witness run standalone, exactly as the orchestrator runs it |
| `trtllm-modeling-v2 perf {campaign,table} ...` | perf tooling outside a run: inspect a campaign, print a curve table |
| `trtllm-modeling-v2 tests {affected,compare} ...` | the test-result cache: the tests a change can affect, and the regressions against the base — see [Tests after a shared change](#tests-after-a-shared-change) |
| `trtllm-modeling-v2 spool {submit,launch,wait,status,cancel} ...` | jobs in a compute session's spool (the same program as `infra/substrate/spoolctl.py`, which a shell without the CLI runs by path) |
| `trtllm-modeling-v2 intake {derive,check,confirm,compile} ...` | Step 1 (INTAKE.md): derive a user-facing `task.yaml` from a checkpoint, check it against its checkpoint, confirm it (recorded in `<task>.confirmed`), compile it with a `site.yaml` |
| `trtllm-modeling-v2 session {render,status,requests,submission,allocate,ack,watchdog} ...` | compute sessions — see [Execution substrate](#execution-substrate) |
| `trtllm-modeling-v2 driver --workspace W {wait,pending,status,claim,release,prompt,answer,instructions,reload}` | the driver side of `--agent-transport external`, and with `wait --sessions-only` the operator loop of an in-process run (see [Execution substrate](#execution-substrate)); `reload` loads changed flow code into the running orchestrator — see [Reloading and relaunching](#reloading-and-relaunching) |

## Running the turns in process

By default (`--agent-transport inprocess`) the orchestrator runs every role
turn itself, on an `AgentLayer` (`agent_flow`, Claude Agent SDK), which starts
one `claude` CLI per turn in the workspace. No driver session is needed. The
orchestrator still writes `dispatch/orchestrator.heartbeat`, and
`orchestrator.exit` when it exits, so a driver watching the run's sessions
reads a crash as an exit.

On a Slurm cluster, launch it from the login node:

- **From a plain terminal or a tmux session, not from inside another agent's
  session.** A turn started there inherits that session's sandbox, and its own
  Bash tool cannot start the sandbox the managed policy requires
  (`... srt-mux-*.sock ... EPERM`): the model then *reports* that Bash failed,
  and the turn ends as if it had worked.
- **Logged in.** The CLI authenticates from the launching shell's `HOME` (and
  `CLAUDE_CONFIG_DIR`); run `claude` once there.
- **With a CLI the organization accepts.** The SDK bundles its own `claude`,
  which can be older than the organization's minimum version; then every turn
  is refused at start-up. `AGENT_FLOW_CLAUDE_CLI` names the binary every layer
  launches. Without it a `claude` on `PATH` newer than the bundled one is used.
- **On the model and effort you mean.** Without an `agents` block every role
  runs on Claude Code at its default model (`claude-opus-5-5`) and the
  backend's maximum effort. Name them for every role:

```yaml
agents:
  defaults: {backend: claude-code, model: claude-opus-5-5, reasoning_effort: xhigh}
```

```bash
tmux new -s mv2                                # on the login node
export AGENT_FLOW_CLAUDE_CLI="$(command -v claude)"
trtllm-modeling-v2 --task task.yaml --site site.yaml --workspace "$WS" --dry-run
trtllm-modeling-v2 --task task.yaml --site site.yaml --workspace "$WS" 2>&1 | tee -a "$WS.log"
```

**Preflight proves the turns work before the run spends anything.** Under the
in-process transport, preflight (a `--dry-run` too) runs one tiny turn on every
distinct agent the roles resolve to (`run/agent_probe.py`): the turn must run
one command through its Bash tool that writes a nonce, the nonce's SHA-256 and
`git -C <trtllm_repo_path> rev-parse HEAD` into
`gates/agent-probe/<n>/probe.out`. Preflight judges the file, never the model's
words: a model that only says it ran the command cannot write a hash it never
computed. A failure stops the run before any planner turn and says, per agent,
what the turn raised or said, the CLI's stderr, where the turn's files are, and
what to do: `AGENT_FLOW_CLAUDE_CLI` for a refused CLI, a plain terminal for a
sandbox `EPERM`, a login for an authentication failure, or
`--agent-transport external`. A pass is kept in `gates/agent-probe/pass.json`,
and a resumed run (a reload, a relaunch, the run after a `--dry-run`) stands on
it for 12 h while the agents, the CLI and its version, the host, the home it
authenticates from, whether this process runs inside another agent's session,
and the checkout path are what they were; the journal says `agent probe passed`
or `agent probe not run`, and why. `MV2FLOW_SKIP_AGENT_PROBE=1` skips it, for
when the probe itself is what is broken.

**Watching a turn.** Each in-process turn leaves `turns/<id>.*`, written as it
goes: `request.json` (role, task, agent, process, start, bound), `system.md`,
`prompt.md`, `activity.log` (every tool call with its input, text and thinking
block as it streams, and the CLI's stderr), then `result.md` and, last,
`done` (`answered`, `raised`, `timed out` or `interrupted`). `tail -f
"$WS"/turns/*.activity.log` follows them all. During the planned run,
`driver status` lists the turns in flight as
`orchestrator holds: turn <role> <task> <age>`.

**Bounds.** An in-process turn has its role's timeout (see `--agent-turn-timeout`
above). Past it the turn is stopped, its CLI with it, and it raises
`TurnTimeout`, as an external turn nobody answered does: a task's turn is
retried once as an infrastructure failure, then fails to the planner.

## Running where the orchestrator cannot authenticate an agent

On a sandboxed host the spawned CLI cannot write its credential lock, and the
turn fails with `EROFS ... credentials.lock` reported as
`stop_reason='stop_sequence'` and **zero tokens** — shaped exactly like a model
refusal when in fact no agent started.

Copying the credential store to a writable home does not fix it for long: the
copy shares a single-use refresh token with the parent session, and one of
the two loses within minutes. Use `--agent-transport external`, so an
already-authenticated session runs the turns.

External transport (`infra/transport.py`) moves the **turn** and nothing else. The
orchestrator writes `dispatch/<id>.system.md`, `dispatch/<id>.prompt.md` and
`dispatch/<id>.request.json`, then blocks until a driver writes
`dispatch/<id>.result.md` followed by `dispatch/<id>.done`. The driver can be a
human, another session or a service; it is trusted with nothing, because every
*decision* still arrives through `collect_decision_file` — validated against the
role's closed enum — and every *verdict* still
comes from a gate the orchestrator ran itself.

The one thing a driver must get right: **the role's system prompt travels with
the turn**. A driver that delivers only the user prompt is running a different
agent, and its answer still looks plausible. When `task.yaml` routes a role to
a particular agent, the request names it (`agent`); otherwise the driver runs
the turn on its own model and effort.

## Execution substrate

`environment.kind` is `local` (this host, the devices preflight claims) or
`session_spool`. On a Slurm/pyxis cluster a run stands on one or more **compute sessions**:
allocations whose main process is the spool scheduler
(`infra/substrate/scheduler.py`, started by `spool_watcher.sh`). Work is a script
renamed into `<spool>/pending/`; the receipt is `done/<name>.rc`, written last,
with the verdict beside it in `done/<name>.json`. The orchestrator submits
through `infra/substrate/spoolctl.py`; agents and operators through
`infra/substrate/spoolctl.py submit`.

**Declare the sessions, render their entries.** `environment.sessions` in
`task.yaml` names each session's spool, role (`gates`, `work`, `any`), node
count, GPUs and time limit. `trtllm-modeling-v2 session render --task
task.yaml --name a` writes the session's entry script from that block, so no
entry script is hand-edited; `session submission` prints the broker call that
starts it, `session status` shows every session.

**The scheduler** gives each job what its header asks for
(`spoolctl.py submit --gpus N|all --wall 2h --tree <checkout> <name> '<cmd>'`):

- one priority queue; whole-node jobs go before slot jobs, and a waiting
  multi-device job holds the devices it needs, so nothing is starved. Within
  those, jobs run in the order they were queued: by the pending file's
  modification time, and within one second of it by the `#spool-queued:` line
  the submitter stamped to the microsecond (a shared filesystem may keep
  modification times to the second), never by name;
- device slots, exported as `CUDA_VISIBLE_DEVICES`. Under any scheduler
  preflight claims no devices and no prompt names one, and the files a gate's
  job reads are written under the workspace, on the shared filesystem;
- a wall limit counted from the job's start, a resident-memory limit, and an
  optional host-memory pressure floor;
- a GPU job starts only on devices whose memory the last job released: one
  `nvidia-smi` reading as it is about to start hands out devices with at most
  `device_max_used_mib` (1024) in use first and holds the job for the rest; a
  device still over it after `device_release_wait` (5 min) is quarantined until
  a reading finds it released;
- the allocation's MPI context (`PMI*`, `PMIX*`, `OMPI*`, `SLURM*`, ...) removed
  from every job, core dumps off. Once a job is over, processes that escaped
  its session are found by their marker (spool, watcher and job) and stopped;
  exited orphans are collected only when the scheduler adopts them (PID 1 or
  a child subreaper);
- names checked against the requester's prefix (each external turn gets one),
  and earlier results of a reused name archived rather than deleted;
- **the safe-kill rule**: it stops its own jobs (limits, `spoolctl.py cancel`)
  but never signals a process that still carries the allocation's MPI context;
  such a job is marked failed and its devices are quarantined until it exits.

**`--tree <checkout>`** runs a job with that checkout as its working directory
and first `PYTHONPATH` entry. Every job carries a resolve hook that records
which checkout each process imported; a result whose ranks disagree, or that
did not import the declared tree, is refused (`reason=mixed_checkout`). Gates
declare `trtllm_repo_path` as their tree. Questions about the target's git
state are asked of the checkout from the orchestrator's own process, never
through a spool job (`infra/vcs.py`).

**Native rebuilds** go through `infra/scripts/incremental_build.sh`: it announces the
rebuild on every session (their schedulers hold new GPU jobs against that
tree, and against every tree whose native build resolves into it: the
snapshots and the task and tune worktrees link `tensorrt_llm/libs` and the
bindings to the checkout's), gives every installed library a fresh inode so
mapped copies are never rewritten, builds, checks the import and retracts the
announcement.

**No hang needs a human.** A gate's deadline counts from its start; a gate
whose session is lost is withdrawn and re-run once a session is back (see
[the gates](#gate-workloads-floors-and-findings)); a job that prints a
traceback and then goes silent is marked `stalled`; an engine waits for the
host memory and `/dev/shm` it declares (the VMM release lag) and is refused
with the reason if it never comes back; `trtllm-modeling-v2 session watchdog` returns the
moment a session is stale, ending without a successor, or quarantined.

**A lost allocation.** When Slurm requeues a session onto another node, or an
allocation ends or fails with jobs running, those jobs go with it. The next
scheduler on the spool (the requeued start, or a successor) finds them in
`running/` and queues each again with its place in the queue, once unless the
job says otherwise (`spoolctl.py submit --requeue N`, header `requeue=N`; the
scheduler's `--requeue` sets the session's default). A job that may not run
again (`requeue=0`, or no re-runs left) or whose cancel was requested is
closed out `reason=node_lost` (rc 218). The receipt's `attempts` lists every
allocation the job ran on; `spoolctl.py submit` and `wait` say on stderr when
a job is queued again and wait for that run, and `spoolctl.py status` lists
such jobs. A job whose processes outlived only their watcher, on this host, is
closed out `orphaned` rather than run beside itself.

**Hand-over.** A session with a `time_limit` writes
`<spool>/handover/request.json` `handover_lead` before its end and stops
admitting jobs that would not finish, so they wait in the spool for the
successor that watches the same spool; once the successor is up the old
scheduler retires. The successor is acquired by the driver; the orchestrator
never calls the cluster's scheduler:

1. `driver wait` returns `SESSION_REQUEST` (or `SESSION_ENDING` when no request
   was written);
2. `trtllm-modeling-v2 session allocate --task <task.yaml> --name <session>`
   prints the broker call, with `dependency=afterany:<job>` on the allocation
   the session's **live** watcher holds (the one that ends last, if two
   overlap), and says why. With no live watcher it prints the call without a
   dependency and says so; when that allocation already has a successor
   acknowledged it says that too, since a second one would start beside it;
3. submit it, then run the `session ack ... --job <new id> --after <job>` it
   printed.

The hand-over state is keyed by the allocation it is about. An ack is
`<spool>/handover/acks/<allocation>.json`: written before the request, it
answers the request when the scheduler writes it, and an early ack for the next
allocation is a file of its own, so it never replaces the one that answered
this request. A request stands only while the watcher that wrote it
heart-beats and no successor is up: when the successor starts it moves its
predecessor's request and acks to `handover/archive/` (its `watcher.status`
says so), and `driver wait` archives any request no live watcher stands
behind (an ended allocation's, or an earlier start of a requeued one) and says
so on a `note:` line after its event; `session requests` lists those as
ignored. A watcher whose request left the disk while it still needs a
successor writes it again.

A job sent to a spool that will not start it soon (in hand-over, ending before
the job's wall limit, or with no live watcher) gets a `warning:` from
`spoolctl.py submit` naming the spools of the same kind (`gates-*`, `work-*`,
...: same directory, name without its trailing number) that can take it now;
`spoolctl.py submit --pick` sends it to the one that can start it soonest, and
the orchestrator's gate pool passes over such a session the same way
(`infra/substrate/placement.py`).

**The operator loop of an in-process run.** With `--agent-transport inprocess`
no turn reaches a driver, but the sessions still need their hand-overs and a
watchdog, and someone must notice the orchestrator stop. Run, in a session that
holds the broker tools and on the orchestrator's host:

```bash
trtllm-modeling-v2 driver --workspace <ws> wait --sessions-only
```

It never returns `NEW` and returns everything else: `ORCHESTRATOR_EXITED`
(read its log; relaunch it unless the run closed), `RELOADED`,
`RELOAD_OVERDUE`, `RELOAD_FAILED`, `SESSION_REQUEST` (the hand-over above),
`SESSION_STALE`, `SESSION_ENDING`, `SESSION_QUARANTINED` (look at that
session), or `TIMEOUT` (call it again). Its status block shows the turns in
flight as what the orchestrator holds (`turn <role> <what>`). Such an
orchestrator heart-beats like an external one, so a crash reads as
`ORCHESTRATOR_EXITED` from any host. Without a fresh heartbeat (an
orchestrator from before heartbeats ran in process), `wait` reads the run's
ledger (`inflight.json`) and the workspace lock (`.orchestrator.lock`, held for
the whole run): a held lock is a running orchestrator, a run that closed says
so in the ledger, and a free lock on the ledger's own host is one that stopped
or crashed; on another host `driver status` says `unknown` with the host to run
it on.

**Multi-node.** A session with `nodes: 2` runs one scheduler per node;
`infra/scripts/mn/mn_mpirun.sh` launches one MPI job across them through the
schedulers, with no sshd or srun in the container (`infra/scripts/mn/README.md`).

**Executors.** `infra/execution/` turns a command into a result under the resolved
environment: `commands.py` (local), `spool.py` (a spool job), and `turns.py`
(a role turn, in process or through `infra/transport.py`). A gate goes to the
gate sessions; one none of them would start at once goes to another session
that would (devices free, nothing queued), and failing that to a tune dev
session (`perf.dev_spools`, less any in `perf.spools`) while nothing runs or
waits there. `check_capacity` counts by the same rules how many more checks
would start at once (file reads only; never under 2 in flight nor over 32, and
8 when no session can be read), and `capacity.CheckAdmission` holds a check
until one would.

**The external driver.** With `--agent-transport external`, the driver session
runs `trtllm-modeling-v2 driver --workspace <ws> wait`, which returns `NEW <id>`
(a worker's within an in-flight cap of 16, the orchestrator's default task concurrency; the planner's turn and the critic's read at once, ahead of any worker's and beside the cap), `ORCHESTRATOR_EXITED`, `RELOADED`,
`RELOAD_OVERDUE`, `RELOAD_FAILED`, `SESSION_REQUEST` or a
session alert; `claim`, `prompt <id>`, `answer` and `status` complete the loop,
`instructions` renders the standing `TURN_INSTRUCTIONS.md` from `task.yaml`, and
`reload` asks the orchestrator to reload itself (next section).
A relaunched orchestrator adopts every unanswered turn rather than asking again,
and every answered turn of a task that had no outcome yet: the task's first
dispatch in the new process collects the decision where that turn wrote it,
and `driver status` marks the turn `ADOPTED`. A turn is consumed once its task is done, failed or
abandoned, so a task the planner reopens is asked afresh; the planner's,
critic's and improver's answered turns are never adopted.

## Reloading and relaunching

The orchestrator imports the flow's code once, when it starts: a fix to the
loop, a stage, a prompt or the contract reaches a running run only through a
new process. What an agent runs (the gate CLI, `spoolctl.py`) and the spool
jobs a tune campaign submits are processes of their own, and read the code as
it is when they start.

**Reload, not kill.** `trtllm-modeling-v2 driver --workspace <ws> reload` writes
`reload.request.json`; the orchestrator notices within its poll interval
(5 s) and drains (`run/drain.py`):

- it starts no task, no integration, no contract check and no planner turn;
- what is in flight runs to its end: an integration (and a worktree being
  made), a contract check that started, and the planner's turn or the critic's
  read. A check that would start meanwhile waits for the new process;
- worker turns on the external transport go on: the new process adopts every
  turn the old one asked, answered or not;
- on the in-process transport a turn dies with its process, so the drain also
  waits for every turn in flight, a tune campaign's included.

Once nothing is left, the orchestrator checkpoints, journals a `reload` event
with what it waited for and for how long, and re-executes itself
(`os.execve` of its own interpreter command line, under the environment it was
started with): same pid, same log, same arguments but for `--clean`. The new
process resumes from the checkpoint; when its loop runs, `reload.json` says
`reloaded` and `driver wait` returns `RELOADED`. Its preflight, like that of
any resumed run, does not run the substrate gate again when the last pass
(`gates/substrate/pass.json`) is under 12 h old and was recorded for the same
resolved environment, checkpoint, runtime block, options file and checkout
path; the journal says `substrate gate not run` or `substrate gate passed`,
and why. Delete that file to have the next start prove the substrate again.

The drain waits `--timeout` seconds (default 1800). Past it, it journals what
it still waits for, `driver wait` returns `RELOAD_OVERDUE` once, and it keeps
waiting; `driver reload --force` re-executes at the deadline without what is
left (a cut-short check runs again in the new process), and `driver reload
--cancel` withdraws the request. `driver status` shows a request not yet
taken up, a drain and what it waits for, and what the orchestrator holds right
now. A re-exec that fails (`RELOAD_FAILED`) leaves the run going on in the old
process, on the old code. The planned loop is the only place a reload is
answered: a request made during preflight waits for it, and one left when the
process exits is answered by the next start. The tune stage's own measurements
are not drained: a campaign resumes from its ledger.

**What a stop cost.** Every start reads `inflight.json`, the ledger the previous
process kept of what it held, and journals `restart (<how>)`: `reload` (nothing
cut short), `hard-stop` (the checks, integrations, worktrees and in-process
turns it cut short; the check jobs it left queued or running in the spools,
which nothing reads any more while their checks run again; the tasks set back
to pending; and on the external transport the turns it left: still with the
driver, answered and never read, or answered for a task with no outcome yet),
`closed` (a run re-entered after it closed) or `unrecorded` (a flow from before
the ledger). Each start is a line in `restarts.jsonl`.

**Relaunch** only when there is no orchestrator to reload (`driver status` says
`exited`): run the same command again. It adopts every turn its predecessor
asked, the checks a hard stop cut short run again, and an integration a stop
cut short is integrated again from the task's branch.

## Commits

Agents commit only through `<workspace>/bin/commit.sh` (`infra/committer.py`). It
commits and signs off as the repository's configured user — so the DCO
sign-off email always equals the author email — takes explicit repo-relative
file paths only, and refuses untracked symlinks (a worktree's link to the
native build), sandbox placeholders (`.mcp.json`, `.claude/`), the run's own
workspace and anything outside `commit_policy.allowed_paths`. The CLI refuses
to start without an identity, after one trial commit in a throwaway repository
proves this environment's git writes what the rules require.

The identity is the target repository's configured `user.name` /
`user.email`, or `MV2FLOW_GIT_NAME` and `MV2FLOW_GIT_EMAIL` when both are
set. The optional `commit_policy` block in `task.yaml` sets:
- `allowed_paths` (default `tensorrt_llm/ tests/ cpp/ examples/`);
- `allow_co_authors` (default false);
- `base_ref`, the merge base delivery audits from (default `origin/main`,
  `upstream/main` or `main`).

A refused commit prints `commit.sh: REFUSED` and the reason; see
TROUBLESHOOTING.md. Delivery runs the same rules over the branch: every commit
since the merge base that touches the run's allowed paths must be signed off by
its author, with no co-author, symlink or placeholder (`git:commit-hygiene`).

## Source embargo

`source_embargo` in `task.yaml` lists paths no role may read. Entries take
three forms:
- a repo-relative path, for an implementation of this exact checkpoint in the
  tree;
- an absolute path outside the repo, for a sibling engine's checkout (a
  directory embargoes its whole tree);
- `package:<path>`, inside this workflow package, for its own worked
  examples, e.g. `package:examples/<model>.md`.

The schema refuses an entry that does not exist (`spec/task_schema.py`). The embargo is
a rule, not a tool hook: every role prompt states it and lists the paths
(`agents/prompts/_common.py`, `source_embargo_block`), and the driver's turn instructions
repeat it. Running the other implementation, as an accuracy oracle, stays
allowed.

No blanket rule keeps agents out of TensorRT-LLM's per-model definitions: the
zoo is an ordinary read. What carries the duty instead is the `modeling-core`
invariants — a copied forward cannot pass the closed-vocabulary audit, is not
flat, and leans on shared helpers a target may not have — plus the rule kept
verbatim in all three writing roles: *read to understand an interface, never
to copy a structure*.

## Tuning

Onboarding makes a target *correct*; tuning makes it *fast* without making it
less correct. Perf is measured, never a release gate — but every kept change
passes the accuracy gate, and every number traces to a label dir.

With `perf.enabled: true` the planner must put one `tune` task in the graph.
The orchestrator holds it back until every inner check of the first unit
(the core's boot and accuracy) has passed, on the current target or an
earlier one, with no failure since, then starts it on whatever devices the
build work leaves idle. The task pins the commit the target is at when it
starts and works in its own worktrees, so feature tasks landing meanwhile
never reach a measurement, and an integration that leaves the core's passes
stale does not hold it back. When it ends, its report names the best branch;
merging that branch into a target that has moved since is an `integrate`
task the planner adds. A failed tune task is reported to the planner with
what failed and where its job log is, not as a failed run: delivery does not
wait on performance. Nor does correctness: final reviews and the tasks that
claim delivery-tier checks never depend on the tune task or on integrating
its branch, the delivery tier runs beside the campaign, and integrating its
best branch is a later task whose checks run again on the merged tree (see
[The pipeline](#the-pipeline)).

```bash
trtllm-modeling-v2 tune --task task.yaml --workspace W        # a target onboarded elsewhere, or re-enter a closed run
trtllm-modeling-v2 perf campaign status --spec tune.yaml      # tooling: candidates and verdicts
trtllm-modeling-v2 perf table --baseline A A B C              #          a curve table of label dirs
trtllm-modeling-v2 perf ab --a A --b B --a A2                  #          judge an A/B as the campaign does
```

The campaign has two layers (`tune/`):

| layer | who | what moves | judged by |
|---|---|---|---|
| **ground truth** | Python | a perf smoke, then the baseline (`modeling_v2: require`), its identity repeats and the stock reference (`"off"`) at once over the perf nodes; the repeats' same-host spread = the noise floor | -- |
| **campaign** (`campaign.py`, `stage.py`) | `tuner` turns | the forward: fusion, ordering, overlap, kernel source; one candidate = one worktree = one hypothesis | e2e A/B + the keep gate, in Python |

Both measure at one serving config. Without `perf.identity_configs` it is
derived from the workload (`serving_config.py`): `max_batch_size` is the
highest concurrency, `max_seq_len` is ISL + OSL (+ the draft length),
`max_num_tokens` holds a whole prompt plus one token per other request
(a power of two in [8192, 32768]), chunked prefill is on only when a prompt
does not fit it, and CUDA graphs capture every batch size up to the maximum,
with padding. A target's own switches are tuner hypotheses, not defaults.

- **The keep rule** (`records.compare`): a WIN is at least `max(min_gain,
  noise_factor x spread)` better on some headline metric (tok/s/user at the
  lowest concurrency, tok/s/GPU at the highest) and no worse than
  `max(1%, noise_factor x spread)` on any. The spread is measured on the tree
  under test: the same-host controls of the current best tree (for the
  baseline tree, also the ground runs of a member that one host ran twice),
  per site, the widest any host's controls read, one spread for both metrics
  of a site. A floor measured on an older
  best is never used: a keep can make a point far noisier than it was on the
  tree before. Every measurement and decision is appended to
  `perf_ledger.jsonl`.
- **tok/s/user** (`tune/bench.py`). A point is one closed loop: the client
  sends `concurrency` requests at once, then one more as each finishes. How
  the server admits that first wave sets how every later one arrives, all at
  once again or staggered, and the two patterns trade TTFT against TPOT at the
  same throughput and the same mean end-to-end latency. `tok_s_user` is the
  output tokens over the end-to-end latency, both summed over every request
  of the measured run, which Little's law ties to the point's throughput
  whichever pattern a run falls into; at concurrency 1 it agrees with the
  mean of each request's rate. Each summary row names that basis
  (`tok_s_user_basis`) and keeps that mean, benchmark_serving's figure, as
  `tok_s_user_per_request`: it grows with the spread of latencies a pattern
  sets, and at an attention-DP point reads a blocked run a few percent below
  a staggered one. A discarded warm-up wave would not settle the pattern (the
  measured run starts its own wave), nor would a window past the first wave
  (the pattern lasts through every wave). A summary whose rows name no basis
  is restated on this one when it is read (`records.py`).
- **Ground truth over every perf node.** The baseline, its
  `perf.noise_repeats` repeats and the stock reference are measured at once.
  Each perf config (member) is a run of its own: every idle node takes its
  share of the runs left as one job, never more runs than one label has, and
  first the members its host has not run, so a member's repeats run on other
  hosts than its baseline, beside it. While the repeats have members a node
  has not run, it takes one run of each: the runs one host made of every
  member are a control of that host (the label itself, or `t-ground-<host>-<n>`,
  links to the runs it takes), and the campaign adopts each as that host's
  first control. A label measured across nodes records the host of each
  member (`hosts` in its `meta.json`) and is no host's control. The noise
  floor a keep is judged under stays a same-host one: what a member's runs on
  one host spread by; with no member run twice on one node, each node's
  controls (A, A') set it. The spread across hosts is recorded beside it
  (`cross_host` in the ledger's `noise_floor`, `state.tune.cross_host_spread`,
  and the journal's `[tune] noise floor` line) and judges nothing. The jobs in
  flight are checkpointed with their runs (`state.tune.queued_runs`), so a
  restart adopts them; a run that fails is measured again from where its job
  stopped. The wrap-up's `t-final` spreads the same way.
- **A, B, A' on one idle node.** Every measurement starts on an idle node:
  nothing runs on it, whoever's job, and the last job there ended
  `perf.campaign.idle_settle_s` (default 120 s) ago: a point measured right
  after another job reads low by more than the floor. Per best tree, each
  perf node measures a control (A) before the first candidate (B) it measures
  and a second (A') right after; a candidate is judged against the mean of its
  node's controls. The measurement of a keep is the new best's first control
  on its node.
- **Measured again before it counts.** A LOSS whose every loss is within twice
  its loss floor is what one disturbed run looks like: the candidate is
  measured once more on the same node, then a control (B', A''), and judged
  on the means of all of them. A node that leaves before its controls are in
  sets the candidate's measurement aside (`<label>.unpaired-<time>`) and it is
  measured again where a pair can be made.
- **Members a candidate does not change** (`tune/inert.py`). The tuner's
  decision names `changed_members` (the perf configs whose engine runs the
  change) and a `path_marker` (a line the change prints where its path is
  taken). A member whose server log lacks the marker while another member's
  has it, and whose engine reads as the control's (its `MV2FLOW_ENGINE`
  records and pool lines; without any, the decision's `changed_members` must
  leave it out), is judged nowhere: a loss there is measured again, and a
  loss repeated there is journaled as the host's noise (`host_noise` in the
  ledger), never held against the candidate. A measurement that holds none
  of the `changed_members` (a re-measure of the members a first round lost
  at) gives the marker nowhere to print: there the list itself leaves a
  member out, provided its engine reads as the control's; without those
  readings nothing is inert. `trtllm-modeling-v2 perf ab --a A --b B --a A2
  [--changed-members M,...] [--path-marker T [--marker-log LOG]]` applies the
  same judgment to an A/B a task runs by hand; a `--marker-log` is a log
  outside the measurement where the marker printed (a gate's, at a member
  that runs the change), and counts as the marker printing.
- **The perf smoke.** Before the baseline, the stage serves each distinct
  world size once (the first perf config of that size, concurrency 1, 4
  prompts of 32 output tokens, a 45-minute wall, `perf.campaign.smoke_wall`)
  on the perf sessions and the tree the baseline measures. A launch or
  teardown defect (a config that does not select the target, which the smoke
  reads from the `modeling_v2` mode it served; a multi-rank launch that hangs;
  ranks left on the devices; a device another job holds) ends the task in
  minutes as `tune-failed:smoke`, with the job's log, instead of after a whole
  baseline. Only the substrate's own failures are re-queued. The smoke is
  skipped once the baseline is measured, and with `perf.campaign.smoke: false`.
- **The tuner claims, the orchestrator measures.** A tuner turn implements
  one hypothesis in `W/tune/worktrees/<slug>` (branch `tune/<run>/<slug>`, native
  build linked in), proves it at the entry level (cells, GPU test) and in a
  micro-benchmark (`tune/microbench.py`, CUDA-graph and eager timings with
  per-layer weight rotation), and answers `CANDIDATE`. Python then measures
  the worktree end to end (`#spool: tree=`), applies the keep rule, runs
  the keep gate, and cherry-picks a keep onto `tune/<run>/best`.
- **The speed of light** (`tune/sol.py`). After ground truth, one tuner turn
  derives the SOL ceiling of every headline metric: the work one step
  necessarily does (weight and KV bytes, FLOPs, collectives), priced against
  the hardware. It follows the toolkit's `internal-perf-sol-analysis` skill
  when the session has it installed (`trtllm-agent-toolkit` plugin): hardware
  peaks from its datasheet calculator, latency constants measured with its
  `measure_channels.py` on a compute session, every number from its
  `sol_calc.py`. Without the skill the turn derives the ceilings by hand and
  says so (`method: reduced`). The stage refuses a `sol.yaml` with a ceiling
  below what the baseline already measured (one fix turn, then the campaign
  goes on without ceilings). With ceilings, the roadmap joins the profile
  against SOL and ranks by the gap each item can recover rather than by time;
  standings and `summary.md` show each measurement's % of SOL; and
  `perf.goal.pct_of_sol` stops the campaign once every headline reaches that
  fraction of its ceiling. Reduced ceilings judge that goal too: when
  `sol.yaml` records `published_peaks` and `measured_peaks` (keyed by the
  bound each sets: `memory`, `compute`, `comm`), each ceiling is scaled by
  measured / published for its bound, and otherwise taken at the published
  peaks. The journal, the ledger (`goal_reached`) and `summary.md` name the
  ceiling basis that decided it. `perf.sol.enabled: false` skips the turn.
- **The roadmap** comes from a first tuner turn (profile + code reading ->
  `W/tune/roadmap.yaml`), or from the operator. Candidates run
  `perf.campaign.max_parallel_tuners` at a time; the stage stops after
  `max_consecutive_misses` misses or `max_iterations` candidates. A spent
  roadmap with budget left is refreshed by a roadmap turn on the moved best
  tree (`roadmap-2.yaml`, ...; `max_roadmaps`, default 2), told what was tried.
- **The wall budget.** `perf.campaign.max_hours` (none by default) bounds the
  campaign's wall time, counted from the task's first start: ground truth,
  restarts and retries count. Once it is spent no candidate, rebase,
  member-gated variant or roadmap is issued; the candidates already issued
  (a turn in flight, or one that answered before a restart) are measured and
  judged as usual, and the campaign ends with its best kept tree as if the
  roadmaps were spent. The end says so: `state.tune.ended.by: budget`, the
  ledger's `budget_spent` and `campaign_end`, the task's report to the planner
  and `summary.md`. A budget raised in `task.yaml` before a restart is read as
  it stands; a campaign entered again after it ended (`trtllm-modeling-v2
  tune`) has a budget of its own.
- **Stacked evaluation and batches.** A candidate is measured cherry-picked
  onto the current best tree, against the controls of that tree on the same
  host (hosts differ by more than most candidates move). Winners on one best tree
  wait for each other and are stacked: the compatible set with the largest
  summed gain is measured once and gated once, and must hold every member's
  best single gain. A winner that conflicts with the batch in the forward file
  is sent to a REBASE turn onto the batch commit, and stacks on the new best
  when the batch is kept.
- **A LOSS split by member** (`tune/split.py`). Over several perf configs a
  candidate can lose at a few members and win at the rest. A site is one
  member at one concurrency. When every site that lost shares a parallel mode
  that no site that won is in -- attention DP on or off, an EP, TP or world
  size, the requests a rank runs in a step (low or high concurrency), or two of
  those together; the fewest traits, then the fewest sites -- the candidate is
  `split` instead of reverted. One tuner turn builds its member-gated variant
  (`<slug>-gated`, in a worktree at the tree that was measured): the same
  change behind that condition, read from the target's `mapping` when it is
  built (per step for the requests), so the losing mode keeps the old path.
  The variant is a candidate of its own: it counts against `max_iterations`,
  is measured and gated like any other, and wins only if it loses nowhere and
  every point the candidate won still wins or holds its gain within the loss
  floor. One variant per candidate and none of a variant; a split past the
  budget or the goal is reverted. Losses and wins at one site, or losses no
  mode separates from the wins, stay a LOSS (journaled as not split). The
  `[campaign]` journal lines and the ledger (`candidate_split`,
  `split_variant`, `variant_verdict`, `split_closed`) record each step;
  `perf.campaign.split_variants: false` reverts every split, as does the
  `tune campaign` CLI, which runs no tuner.
- **The keep gate** (`tune/keep_gate.py`) is the contract's accuracy gate,
  run on the tree a keep would make the best. The benchmark, the anchor row
  (`reference_yaml`, `model_key`, `discriminators`), the floor rule and its
  sample count, and the unit's options file, runtime and `benchmark_args` are
  the contract's, and `checks/gates.run_accuracy_gate` builds the command and
  judges what it printed, as it does for the contract.
  `perf.campaign.accuracy` only names the unit (default `core`) and its
  members (default `tp1` and the unit's widest member, attention DP before EP
  before TP among equals); every member must clear its floor. With the members
  run the target's own test suite and the modeling_v2 contract tests
  (`perf.campaign.gate_tests`): candidates each pass their own op tests, a
  stack of them can still break a block test none of them ran. Each part is a
  job on the dev sessions, side by side where their devices allow; a failed
  part stops the others, a run that reached no verdict runs again (once, twice
  for the substrate's own return codes), and only a gate whose every part
  passed keeps. `tune/gates/<key>/keep.json` records the scores, the anchor,
  the floor, the unit, the members and the tree. A campaign whose contract
  gives the unit no anchor refuses to start (`tune-failed:accuracy`) unless
  `perf.campaign.accuracy: {gate: off}` says so; every keep is then journaled
  as UNGATED.
- **Failures that are not the candidate's** — a session that dies with its
  job, a full filesystem — re-queue the measurement (once, twice for the
  substrate's own return codes) instead of reverting the candidate.
- **Resume and attempts.** An orchestrator restart, or a retry of the task,
  runs the campaign again from `state.tune`: its phase, base commit,
  measurements, roadmap, keeps and in-flight jobs. A turn whose answer is on
  disk (the SOL turn, a roadmap, a fix turn, a candidate) is not asked again;
  a rebase cut off by a restart is submitted when its decision names the
  commit it was asked to rebase onto, and asked again otherwise; and the goal
  is judged, on the spec as it stands, before anything new is dispatched. A
  job's failed receipt is never adopted: the label is measured again with
  retries of its own. From the task's second attempt on, job names carry the
  attempt (`tune<run>a2-<label>`), so an attempt's receipts are its own, and
  a job an earlier attempt left pending or running is adopted under its own
  name rather than queued twice. A label on disk is reused only when its
  `meta.json` says it measured the same workload, merged LLM-API config and
  (on the campaign's own tree) commit; anything else is set aside as
  `<label>.stale-<time>` and measured again. A campaign resumed from a state
  that kept one control per host takes it as that host's A: its next judgment
  there waits for an A' of the current best, and the ground floor counts only
  while the baseline tree is the best.
- **Kernel sourcing for perf** adds two tiers to the onboarding ladder:
  FlashInfer *as installed* (a newer FlashInfer's kernel is an upgrade
  candidate, not a source) and Triton / CuTe DSL / CUDA written by the run.
  When several sources serve one op, all are micro-benchmarked and the
  alternatives are recorded in `kernel_choices.json`.
- **Profiling** goes through the substrate: `bench.py --nsys`. The default is
  session mode (`nsys launch` + an external `nsys start/stop` window after
  `--nsys-delay`); `--nsys-mode api` uses TensorRT-LLM's in-process
  `cudaProfilerApi` recipe, whose stop can hang a serving process (see
  TROUBLESHOOTING.md). `tune/nsys_summary.py` attributes per step class
  (decode / mixed / prefill), placing CUDA-graph replays at their launch.

The `perf:` block (validated by `tune/schema.py`):

```yaml
perf:
  enabled: true                     # a `tune` task in the planned run
  identity_configs: [serving.yaml]  # optional: LLM-API yaml(s); derived from the workload when absent
  workload: {isl: 8192, osl: 1024, concurrency: [1, 128]}
  # Or several named perf configs; every candidate is measured at all of them
  # in one job, and is kept only if it loses nowhere and breaks no constraint:
  # configs:
  #   - {name: chat, workload: {isl: 1024, osl: 256, concurrency: [1, 16]},
  #      constraint: {tpot_ms_max: 30}}
  #   - {name: batch, workload: {isl: 8192, osl: 1024, concurrency: [128]},
  #      llm_api_options: {enable_chunked_prefill: true}}
  # goal: {ratio_to_reference: 1.1}  # stop dispatching once every headline is 1.1x stock
  # goal: {pct_of_sol: 0.7}           # ... or 70% of its SOL ceiling (reduced ones at measured peaks)
  sol: {enabled: true, gpu: gb200}  # the SOL turn; `gpu` is the part the calculator prices
  reference: stock                  # stock = modeling_v2: "off", same configs; or none
  spools: [<whole-node perf sessions>]
  dev_spools: [<1-GPU sessions for tuner jobs>]
  env_script: <sourced by every perf job>
  noise_repeats: 2                  # repeats of the baseline, run beside it on other perf nodes
  campaign: {min_gain: 0.01, noise_factor: 3.0, max_parallel_tuners: 4, max_iterations: 12, max_consecutive_misses: 3,
             smoke: true, smoke_wall: 45m,  # the perf smoke before the baseline
             idle_settle_s: 120,  # a measurement starts on a node idle this long
             split_variants: true,  # a LOSS split by member asks for one member-gated variant
             max_hours: 24,  # the campaign's wall budget; omit it for none
             accuracy: {unit: core, members: [tp1, dep4]}}  # the keep gate's unit and members; or {gate: off}
  source_embargo: [<the checkpoint's in-tree implementation, its kernels and ops>]
```

Skill: `agents/skills/trtllm-modeling-v2-perf-tuning/` (campaign discipline, the
candidate protocol, kernel sourcing, graph safety; `references/nsys.md`,
`references/modeling-iterations.md`).

## Lessons and the improver

The improver turns what went wrong into lessons the next worker is shown, so
a problem that cost hours once costs minutes the second time, and into flow
issues and patch drafts, so a defect of the workflow costs nothing the third
time. It runs beside the run, on a thread of its own, and is woken by:

| wake | from |
|---|---|
| a task ended without finishing: every `FAILED_RECEIPT`, a worker with no decision, `BLOCKED` or `INTEGRATION_FAILED`, the same work failing again | the task's thread |
| a kernel task certifies again an entry a task of this run certified | `graph.yaml` |
| a required task has been ready to start for 30 minutes and has not started (a `tune` campaign waits for the core by design) | `graph.yaml` |
| a planner revision left the plan as it was | `graph.yaml` |
| a flow issue the driver or a worker reported | `flow_issues.reported.jsonl` |
| a check's verdict was void | `contract_status.jsonl` |
| the run closed, whatever the verdict | the close |

A watcher reads those files once a minute while the planned run is under way.
Wakes are batched: the first waits a minute for others to join it, and one
improver turn at most is in flight, so no task, check or planner turn waits
for it; each wake is read once. What woke it is kept in `improve/wakes.json`
until a turn has read it, so a restart loses none. The run's close is the last
turn, and the close waits for it.

The driver and the workers report a flow issue by appending one JSON object
to `flow_issues.reported.jsonl` in the workspace (`agents/lessons.py`
`report_flow_issue`): `symptom` (required), `by`, `evidence` (paths),
`impact`, `suspected` (`<file>:<function>`) and `proposed_fix`. A line that is
not JSON is read as its symptom.

It writes `lessons.proposed.yaml`; Python keeps a lesson only if it has a
valid scope and kind, a trigger and a lesson within 600 characters, and
evidence that exists in the workspace (`agents/lessons.py`). Each lesson has a scope:

| scope | shown to |
|---|---|
| `all` | every role |
| `role:<role>` | one role, e.g. `role:smith` |
| `host:<site>` | every run on this cluster (the host name without its node number) |
| `family:<family>` | every run on this model family |

The store is one YAML file per scope under `--lessons-dir` (default
`$MV2FLOW_LESSONS_DIR`, else `~/.trtllm_modeling_v2/lessons`), shared by
every run that names it and written under a file lock. Seeing a lesson again
refreshes it; each scope keeps its 30 most recently seen. Every dispatch of
every role carries the lessons in its scopes that are not retired, most
recent first, so a lesson learned from one failed task reaches the next task
of the same run.

Each lesson also has a kind. A `domain` lesson is about the model, its
kernels or the site. A `workflow_defect` lesson works around this workflow's
own code (the orchestrator, a gate, the bench, the spool, integration), and
files a flow issue: the symptom, its evidence, the suspected
`<file>:<function>` in the workflow and a proposed fix. Python validates it
and writes it to `flow_issues.yaml` in the workspace, one entry per lesson,
keeping any fix the operator recorded there, so the operator reads a fix list
rather than only workarounds. A lesson may name the commit that fixed it
(`fixed_by`). Once that commit is an ancestor of the commit the running flow
code was imported from, the lesson is retired: it stays in the store, is
journaled once and listed to the improver, and no worker is shown it. Without
git, or for a commit the checkout does not have, it stays shown. A retired
lesson proposed again without `fixed_by` is reopened, since its symptom came
back on code that has the fix.

A flow issue may carry a patch draft: a unified diff, made as `git diff` makes
it from the top of the checkout the workflow runs from, that fixes the
workflow package and adds the failing scenario as a `test_*.py` under the
workflow's tests. The improver makes it in a copy of that checkout (`git
archive` of the running commit under `improve/scratch/`), never in the
checkout the run executes. Python keeps it only when it applies cleanly to the
running commit (checked in an index of its own, so the checkout is not
touched) and changes nothing but the package and its tests; it files it as
`improve/patches/<lesson id>.diff` and lists it, with the commit it applies
to, in the issue. A refused patch costs the issue only its patch, and the
refusal is shown to the next improver turn. The improver checks its proposal
before ending its turn with the command its request names
(`... lessons.main() check-proposal <workspace>`). Nothing applies a patch.

A third kind, `engine_defect`, is a defect of the engine under test, keyed by
what triggers it rather than by the model it was met on: `affects` (a catalog
entry, or an engine file and symbol), `regime` (globs over a vocabulary row's
entry, and the dtypes, words or cell sizes that put a call in it), `detect`
(the test or cell that shows it), and either `upstream_fix` (a commit, or a
pull request as `#<n>`, `<owner>/<repo>#<n>` or its URL) or, while it is
`open`, `disposition` (the fix branch, draft or workaround). Kept across
runs, these lessons are a registry of known defects:

- At preflight a run checks each one in its scopes against the checkout
  under test: a commit that is an ancestor of `HEAD` or that a commit in its
  history names (a cherry-pick with `-x`), a pull request that a commit in its
  history names as `#<n>`. Each fix the tree does not have, or git cannot
  tell, is journaled and added to the host facts every dispatch reads.
- When the survey lands, each vocabulary queue row in a known defect's regime
  is marked (`known_defects` on the row, and the detecting cell among its
  `cells`) before its kernel task is added, and the planner hears it with the
  survey's result, so the fix or workaround is scheduled from the first plan.
- A smith that finds a defect outside its entry ends `FAILED_RECEIPT` with an
  `engine_defect` block (`affects`, `trigger`, `detect`, `proposal`): it is
  registered `open`, and the orchestrator adds the `feature` task that fixes
  it, touching the file `affects` names.

`knowledge: measurement` in `task.yaml` (the default is `production`) leaves
the registry and the lessons of the model's family out of the run, for a run
that measures how the workflow does on a model it has not met; every other
lesson still applies. The improver never writes a lesson that relaxes a
requirement, and never one that carries a model's implementation.
`--no-lessons` turns the improver and the lessons block off, for an
evaluation run that must start from what the workflow ships.

## Feature increments

`task.yaml`'s `features:` is a closed list, because a feature with no skill is
an assembler told to build something nobody wrote down how to build.

| feature | skill | gated by |
|---|---|---|
| `mtp` / `eagle3` / `dflash` / `dspark` | `trtllm-modeling-v2-speculator` | boot, accuracy, **acceptance** |
| `multimodal` | `trtllm-modeling-v2-multimodal` | boot, accuracy on a multimodal benchmark |
| `host_offload` | `trtllm-modeling-v2-host-offload` | boot, accuracy, **residency witness** |
| `disagg` | `trtllm-modeling-v2-disagg` | boot, accuracy, **equivalence witness** |
| `kv_quant` | `trtllm-modeling-v2-kv-quant` | boot, accuracy, **kv_quant witness** |
| `decoder_replay` / `encoder_replay` | `trtllm-modeling-v2-bounded-replay` | boot, accuracy, **replay witness** |

The speculator, `host_offload`, `disagg` and `kv_quant` rows share a property
worth stating plainly: they are **output-identical by construction**, so boot
and accuracy pass whether or not the feature is doing anything. Each therefore
carries a third measurement that can only come out one way if the feature is
real — `acceptance_length` under `=require` vs `=off`, device memory with and
without the offload, token-identical output across the disaggregation hop
(which must stop being identical when the state that crossed is destroyed), or
stored KV bytes per token against a declared figure (plus token identity with
bf16 storage).

The bounded-replay row is the opposite case with the same need. Sliding-window
bounded replay truncates sliding-window state on purpose, so accuracy is a
real check of it — but the truncation is designed to stay inside the accuracy
tolerance, and a unit that implements nothing stays inside it too. Its witness
reads per-request row counters the target prints (`MODELING_V2_WITNESS
key=<int> ...` under `TRTLLM_MODELING_V2_WITNESS=1`) against a window and a
workload the spec declares before assembly.

`dspark` is its own kind and never a spelling of `mtp`: a checkpoint can store
its DSpark block drafter as `mtp.*`, and `--spec_decode_algo=MTP` over those
weights builds the wrong drafter. The schema refuses the two together.

## Worked examples, and what a run is handed

The skills, `runtime_index.yaml`, `kernel_sources.yaml` and
`task.example.yaml` describe the workflow's features without naming a
checkpoint, because a run is handed them and must derive a model's specifics
itself. One model's values for those features (pool formats and byte
arithmetic, replay windows, drafter constants, witness switches) are in
[`examples/`](examples/), as a worked example for operators writing a
`task.yaml`. A run does not materialize `examples/`. A run that onboards that
model to measure the workflow embargoes it as `package:examples/<model>.md`,
which resolves against this package wherever it is installed.

`kernel_sources.yaml` names each kernel source's registration files, and a
source may also claim unnamed `custom_ops/*.py` files by content
(`claims_files_containing`). Tree-local Triton ops, which a model's onboarding
adds one at a time and under no naming convention, therefore classify without
editing the registry.

## What this workflow does not do

- **Performance as a release gate.** A target whose contract holds is
  deliverable whether or not its tune task finished.
- **Enforce the source embargo with a tool.** It is a rule in every role's
  prompt; see [Source embargo](#source-embargo).
- **Turn a delivered branch into a PR.** Lint, copyright headers, PR splitting
  and the CI run are the operator's.

## Tests

```bash
cd agent-flow && pytest tests/workflows/trtllm_modeling_v2/
```

Every test runs on CPU with no model, no GPU and no cluster: the probes and
the shell are injected, and the agent layers are replaced by a fake that
writes the typed decision a real role would have written. A few read the real
checkout, and each reads only what main has:

- `test_delivery.py` verifies that both merged targets,
  `gpt_oss/gpt_oss_120b__sm_103__tp1` and
  `deepseek_v3/r1_0528_nvfp4__sm_103__dep4`, pass every delivery check;
- `test_skills.py` resolves every repository path the skills name;
- `test_kernel_sources.py` and `test_runtime_index.py` hold the registries to
  the tree they describe;
- `test_dspark.py` checks the quickstart `DSPARK` contract against a fixture,
  and against this checkout's quickstart once that has the arm;
- `test_speculation_support.py` reads this checkout's quickstart and LLM API
  for the speculators they handle;
- `test_build_and_test_doc.py` holds `BUILD_AND_TEST.md` to the repository:
  its paths, build flags and test lists.

`test_worked_examples.py` and `test_package_hygiene.py` keep checkpoint
specifics, session placeholders and personal paths out of what the package
ships; `test_package_hygiene.py` also holds the flags table above to the CLI.
