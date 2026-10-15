# trtllm-modeling-v2

Brings one HuggingFace checkpoint to a gated **ModelingV2** target: a flat,
self-contained forward assembled only from catalog entries, trusted through
accuracy gates rather than through shared abstractions.

This workflow is the agent half of modeling_v2.
The code half is already in tree at
`tensorrt_llm/_torch/_experimental/modeling_v2/` — the catalog, the targets,
the router and their tests. What lives here is what builds them: the
orchestrator, the six agent roles, and the ~7,000 lines of skills those agents
load.

```bash
trtllm-modeling-v2 --task task.yaml --workspace workspace/trtllm-modeling-v2
```

Start with `--dry-run`. It runs preflight, resolves the launch environment and
stops before the first agent turn — the cheapest way to find out whether this
host can host this run.

Three companions to this file: **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** is keyed on
symptoms — the verbatim error you saw. **[BUILD_AND_TEST.md](BUILD_AND_TEST.md)** is
keyed on tasks — building, rebuilding, and running at CPU / 1 GPU / multi-GPU /
multi-node scale, collected from the places in TensorRT-LLM that already do each
of them correctly. **[LESSONS.md](LESSONS.md)** is keyed on what real runs got
wrong — each lesson's symptom, its cause, and the mechanism in this workflow
that now prevents it, or the gap that is still open.

**When something breaks, read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) before
you debug the target.** It is keyed on verbatim error text, and the entries
share one property worth knowing in advance: every one of them read as a model
or target defect while it was happening. `MPI_ERR_SPAWN`, a kvCache that
cannot fit a sequence, NVRTC missing `cuda.h`, a 27-point accuracy regression
that was a protocol mismatch — none of those sentences mention the
environment, and all of them were the environment.

## The pipeline

```
preflight  ->  plan  ->  smith x N (concurrent)  ->  assemble x M (serial)  ->  deliver  [->  tune]
  Python     capped         one per                   one per unit             Python     optional:
   only      1 round      catalog entry             core, then features       + 1 turn   perf.enabled
```

`tune` is the performance stage. It runs after delivery when `task.yaml` has
`perf.enabled: true`, or on its own against an already-onboarded target with
`trtllm-modeling-v2 tune --task task.yaml --workspace W` (see
[The tune stage](#the-tune-stage)). It never sits between assemble and
deliver: perf is not a release gate, and a failed tune never blocks delivery.

| stage | who | what it produces |
|---|---|---|
| `preflight` | **nobody** — pure Python | the device claim, the slug, the branch, the resolved launch prefix, the substrate gate, the surface survey |
| `plan` | PlanDrafter ⇄ PlanReviewer | `plan.md`, `vocabulary.yaml` |
| `smith` | one smith per queue item | catalog entries + their GPU tests, committed per entry |
| `assemble` | one assembler per unit | `modeling.py`, `weights.py`, gated |
| `deliver` | Python + one Auditor turn | verification across six trees, then the merge |

Every stage checkpoints before advancing, so a crash resumes at the phase that
was running. Re-running the same command resumes; `--clean` starts over.

## Why it is shaped this way

The predecessor workflow for this job reliably failed the same way: runs spent
their budget deciding *how to launch a job* and re-litigating a plan, and never
reached model onboarding. Four things in this workflow exist to prevent that,
and they are the parts worth not "simplifying" later.

**Preflight is Python, not an agent turn.** The checkpoint parses, the arch
matches the host, the device set is claimed by `memory.free` and a
lone-process probe, `TMPDIR` is sane, and a multi-rank run's NCCL/MPI
substrate answers — all in seconds, before any model is invoked. A broken host
costs seconds instead of a plan phase and a first assembler turn.

**The launch environment is declared once and handed to every agent.**
`task.yaml`'s `environment` block resolves into a single command prefix that
every role's prompt carries verbatim. Agents never construct a launcher
command, never probe for one, and never call a test-runner skill. A prefix
that does not fit is a *symptom to report*, not a problem to solve.

**The plan phase is capped and non-blocking.** `--max-plan-rounds` defaults to
1. Reaching the cap is not a failure: the reviewer's unresolved items are
appended to `plan.md` as `## Open risks` — which the assembler reads — and the
build starts. A plan review never blocks the build.

**Gates are exit codes, not opinions.** There is no `--min-score` and no agent
scoring an artifact out of ten. `trtllm-eval` prints a number, the anchor comes
from TensorRT-LLM's own accuracy references, and the verdict is
`measured >= reference - tolerance`.

## The gates

| gate | when | what |
|---|---|---|
| **boot** | after every assembler turn | minutes. The declared boot workload through the engine driver; keywords are matched on generated text only, and a multimodal target is sent an image |
| **accuracy** | at each unit boundary, every member | the release criterion. `trtllm-eval` against the anchor, with a statistical floor |
| **acceptance** | every unit that keeps a speculator on | `acceptance_length` on ≥ 32 natural-text prompts, paired against the previous verdict |
| **witness** | each feature boot and accuracy cannot see, every member | two legs that differ in exactly one variable; see [Feature increments](#feature-increments) |
| **scheduler-equivalence** | every unit, every member | the overlap scheduler on vs off, greedy at batch 1, token-identical |
| **launch** | last in each unit | a smoke through the unit's own deployment entry (`trtllm-serve`, or its disaggregated launcher) over HTTP |

All of them run with **`modeling_v2=require`**, the LLM-API argument (the
gates export `TRTLLM_MODELING_V2`, and the harness hands it to the LLM API).
Under `auto` a configuration that misses a target's routing criteria silently
gets the built-in implementation, and a number measured that way reads as
modeling_v2's — the single most expensive mistake available in this package.

The **acceptance** gate exists because the first two are blind to speculative
decoding: rejection sampling holds the emitted distribution to the target
model's, so a *miscomputed* draft path produces correct text more slowly. Boot
passes, accuracy passes, only speed moves.

Under `parallel: unified` — the default — boot and accuracy run **once per
member**, where the members are the concrete configurations the one codebase
serves (`members:` in `task.yaml`, default `[tp<W>, tep<W>, dep<W>]`). A
unified target gated at a single topology leaves the expert-parallel and
attention-DP branches unexecuted while its Coverage claims them, so the
orchestrator enforces the member set rather than trusting the prose. Every
verdict names its member (`accuracy[gsm8k]@dep4`) and the unit advances only
when all of them pass. Witnesses run at every member too, cheapest first.
`--dry-run` prints the member set and the resulting gate count before anything
is spent.

## Gate workloads, floors and findings

Every gate's workload is declared in `task.yaml` under `accuracy`, never
sized from the run it judges. Each key also works inside one unit's entry,
for that unit only.

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
  - Per unit, set `min_absolute` with a `floor_source`, or set
    `measured_reference`. A floor with no measured source is warned about.
- **Boot** (`accuracy.boot`, else the workspace `boot.yaml`).
  - `prompts` may carry their own `keywords` and `images`, plus `max_tokens`.
  - `multimodal.prompts` gives the image prompts. Without one, the
    repository's `merlion.png` is sent; it is a git-lfs file, so pull it in
    the tree under test.
- **Witness workloads** (`accuracy.witness.<feature>`).
  - Token-comparing witnesses declare `prompts`/`prompts_file`, `max_tokens`
    and `min_generated_tokens` (default 32).
  - A continuation too short to compare is refused, not passed.
- **Scheduler equivalence** (`accuracy.scheduler_equivalence`).
  - Keys: `units`, `num_prompts`, `max_tokens` (≥ 64), `legs`, and
    `spec_pair: report|agreement|exact`.
  - A target may implement the optional step-trace hook
    (`MV2FLOW_STEP_TRACE_DIR`, see `step_check.py`) to name the first
    divergent step. `step_trace.reference_command` adds a replay through a
    reference implementation.
- **Launch smoke** (`accuracy.launch`).
  - Keys: `num_samples`, `endpoint: chat|completions`, `members: all|first`,
    and `min_accuracy`. Any non-200 response fails.
  - A `disagg` unit must declare `launch.disagg.command`, or set
    `enabled: false` there. The command is a launcher that prints
    `MV2FLOW_ENDPOINT=<url>` once it serves, and stops everything on
    SIGTERM.

Per-prompt baselines and findings are kept in `<workspace>/gate_results.jsonl`.
Every verdict also reaches the evidence ledger.

A unit now runs boot, accuracy, the witness and scheduler equivalence at every
member, then acceptance and the launch smoke. `scheduler_equivalence.units`
and `launch.members: first` narrow that.

**Running one gate by hand.** `trtllm-modeling-v2 gate
{boot,accuracy,acceptance,scheduler-equivalence,launch,witness} --task task.yaml
[--unit U] [--member M] [--tree T] [--results F] [--workdir W]` runs a single
gate or witness exactly as the orchestrator does. With `--tree` it measures
another checkout, which is how a historical defect is replayed on the trees
before and after its fix. Assemblers are pointed at the same command.

**Follow-ups, not built yet:**
- a performance gate: one prefill chunk plus one decode step against the
  previous tree, failing on more than 5% (the engine driver's timing mode
  exists);
- a concurrent, batch > 1 leg beside the batch-1 witness and
  scheduler-equivalence legs.

## Serving envelope and certification

What a target is certified for is declared, not inferred from whatever the
gates happened to run. Two optional `task.yaml` blocks declare it:

- `serving_envelope`:
  - `max_batch_size`, `max_num_tokens` and `max_seq_len`, all required;
  - `cuda_graph`: `false`, `batch_sizes: [...]`, or `padding` with an
    optional bound;
  - `members` in the form `<tp|tep|dep><N>[pp<M>]`. These may include
    multi-node or pipeline-parallel members that no unit gates;
  - optional `max_draft_len`, `gpus_per_node` and `probe_prompt_tokens`.
- `feature_matrix`: named cells, each
  `{name, features, members, evidence: boot|accuracy|witness|acceptance, at: gate|ceiling|{axis: value}}`.

Preflight derives the certification plan and writes `certification.md`. The
plan roles, the smiths and the auditor all certify against it.

- **Consistency.** Every gate's launch tuple must sit inside the envelope, and
  every gated member must be one the envelope claims. Otherwise preflight
  stops and names the inconsistency.
- **Ceiling probes.** Each unit gets one boot at the envelope's limits, at its
  cheapest member, run right after its boot gates. A failed probe sends the
  unit back to the assembler.
- **External evidence.** A matrix cell that no gate or probe reaches is
  *external*. Delivery fails until
  `<workspace>/evidence/<cell>__<member>__<evidence>__<at>.yaml` records
  `verdict: pass`, `tree` and `log` for it.

Without a `serving_envelope`, the envelope is derived from the gate tuples and
marked `DERIVED` wherever it is shown. There are then no probes and no
external cells. To adopt a declared envelope:
- `python -m agent_flow.workflows.trtllm_modeling_v2.envelope migrate task.yaml [--write]` prints (or appends) a block that
  restates the gate hull, so adopting it owes nothing new; widen it from
  there.
- `... envelope plan task.yaml` prints the resulting plan.

The envelope also drives four further checks:

- **Capacity audit** (`capacity_audit.py`).
  - Every target constant that bounds or sizes something must be bound to an
    engine argument on its own line: `# capacity: max_num_tokens`,
    `# capacity: max_batch_size * world_size`, or `# capacity: fixed: <why>`.
    A `CAPACITY_BOUNDS` dict also works.
  - The bound must be at least the envelope's limit.
  - The reviewer is shown the audit. A failure fails delivery, with or
    without a declared envelope.
- **Smith brief** (`smith_brief.py`).
  - Vocabulary steps may declare
    `call_sites: [{site, row_multiple, gathered, eager_only, row_width}]`.
  - Each smith is briefed with the rows every call site reaches at every
    captured batch size.
  - The brief also gives the integer-width probes (2^31 elements, 2^31 / row
    width, 2^24 + 1) that the entry records as hard limits (an `is_valid`
    guard where the op crosses one silently, its `note` otherwise).
  - `python -m agent_flow.workflows.trtllm_modeling_v2.smith_brief --workspace W <request>` renders one brief by hand.
- **`certification_gap`.**
  - Assembler, reviewer and auditor decisions may list
    `certification_gap: [{entry, values, axis, arch, reason}]`.
  - The orchestrator dispatches one verify smith per new surface, bounded by
    `catalog_remediation.max_parallel_verifiers`.
  - The unit then resumes instead of being re-opened.
- **Early checks** (`early_checks.py`).
  - After the plan phase and after each unit, they run as warnings to
    `early_checks.md`: the delivery checks that need no finished target, plus
    capacity failures and owed matrix evidence.
  - The delivery checks covered: test naming, the accuracy test and its
    test-db listing, and whether CI would skip that test.

## Evidence ledger and reports

Every verdict is appended to `<workspace>/ledger.jsonl` together with what it
was measured on:
- the commit and tree object;
- the native build's fingerprint;
- the content of the files it depends on;
- the `TRTLLM_*` switches and their code defaults;
- the unit's slice of the spec;
- each role's backend and model.

Gate verdicts, envelope probes, early checks and delivery checks are all
recorded. A harness run outside the workflow records its own verdict with
`python -m agent_flow.workflows.trtllm_modeling_v2.ledger record <ws> --repo R --gate G --unit U --outcome PASS|FAIL|FINDING`.

- **Freshness** (`freshness.py`).
  - A verdict goes stale when a dependency, the native build, the unit's spec
    or a switch default changes.
  - `python -m agent_flow.workflows.trtllm_modeling_v2.freshness status <ws> --repo R` lists stale and failing verdicts and
    the re-run queue (`rerun_queue.json`).
- **Delivery is bound to one tree.**
  - Delivery re-runs that queue on the delivery tree and its native build.
  - It records a full regression only when every verdict holds there.
    Otherwise the run closes `delivery-failed:evidence`, and it can be
    re-entered once fixed.
  - Lint or a PR split changes the tree. The pre-merge pack therefore carries
    `python -m agent_flow.workflows.trtllm_modeling_v2.freshness check-merge <ws> --repo R --rev <sha>`, to confirm the
    tree being merged is the one that was regressed.
- **Reports are generated, not written.**
  - `progress.json`, `progress.html` and `final-report.md` are rebuilt from
    the ledger, the checkpoint and the journal after every checkpoint.
  - Narrative goes in `progress.overlay.yaml`, whose items may cite verdicts
    as `evidence: {gate, unit, member}`, and in the final report outside its
    `BEGIN/END GENERATED` regions.
  - `python -m agent_flow.workflows.trtllm_modeling_v2.report <ws> [--final-report]` regenerates them.
  - `--check [DOC ...]` exits 1 on any contradiction, for example "no
    finding" beside a FINDING, "all gates pass" beside a failing or stale
    verdict, or a tree that is not the one regressed. Delivery refuses on the
    same contradictions.
  - `--clean` archives the ledger and reports under `<workspace>/archive/`.
- **Records are never overwritten.**
  - Decisions are `decisions/<unit>-<role>-<turn>.yaml`; a reused key gets
    `.2`, `.3` and so on.
  - An external driver must write the decision path its request names,
    verbatim.
  - Journal ordinals are claimed atomically.
- **The catalog is its entry files.**
  - A smith commits one entry file and its test, and nothing else; there is
    no index to write or regenerate.
  - The survey lists the shared entries from `catalog/<category>/*.py`
    themselves, and a target's own ops from its `ops/`.

## The six roles

| role | session | default backend | one run does |
|---|---|---|---|
| `plan_drafter` | persistent | Claude Code | the checkpoint→catalog mapping and the plan |
| `plan_reviewer` | persistent | **Codex** | a paper review of both |
| `smith` | stateless | Claude Code | exactly one catalog entry |
| `assembler` | persistent *within a unit* | Claude Code | one unit: the core, or one feature |
| `reviewer` | stateless | Claude Code | reads one unit's diff against the gate evidence |
| `auditor` | stateless | Claude Code | the runtime-value dump vs the certified columns |

A seventh, `runner`, makes no judgement: it relays a command to a scheduler
only an agent can reach (`environment.kind: slurm_broker`).

The PlanReviewer runs on a different model from the PlanDrafter on purpose: a
reviewer that shares the author's blind spots reviews its own reasoning back at
itself. The plan roles are stateful because they are asked to remember what
they decided; the work roles are not, because this workflow's unit of work is one
entry or one unit, each in its own context.

## Choosing each role's backend and model

The backends above are defaults. An optional `agents` block in `task.yaml`
chooses the backend (`claude-code` or `codex`), model, reasoning effort and
extra MCP servers, for every role (`defaults`) or for one
(`roles.<name>`). It is the shared per-role routing of
`agent_flow.agent_runtime`, the same block `perf-analyze` and `perf-optimize`
read, with the role list and defaults in `roles.py`:

```yaml
agents:
  defaults: {backend: claude-code, model: claude-opus-5-5}
  roles:
    plan_reviewer: {backend: codex, model: gpt-6-astra, reasoning_effort: high}
```

- A `defaults.model` needs a `defaults.backend` beside it. The roles do not
  share a default backend, so a bare model name would also be sent to the
  Codex plan reviewer; the schema refuses it.
- The CLI warns when `plan_reviewer` and `plan_drafter` resolve to the same
  backend and model: the run is legal, but its plan review is not
  cross-model.
- A role on Codex has its typed-decision tool required through the layer's
  backend-neutral `required_tools` policy, because Codex refuses Claude SDK
  hooks. In decision-file mode a Codex role requires no tool; the file is
  validated when it is read.
- Under `--agent-transport external` no session is built here. Each
  `dispatch/<id>.request.json` carries the role's configured backend and model
  as `agent`, and running the turn on a matching model is the driver's job.
- The spec is re-read on every resume (see `_refresh_workspace_spec`), so a
  changed `agents` block applies to every layer the resumed process builds.

Long system prompts are handled by the Claude backend: above
`AGENT_FLOW_SYSTEM_PROMPT_FILE_THRESHOLD` (64 KiB) the prompt is written to
`<workspace>/.agent-flow/system-prompt-<sha>.md` and the CLI is given a pointer
to it, because Linux caps one argv string at 128 KiB. Composed role
prompts are well under the threshold today. The large per-run blocks (survey,
runtime areas, kernel sources) travel in the turn prompt, which is not on
argv.

## Workspace layout

```
<workspace>/
├── task.yaml             the resolved spec, defaults filled in
├── prompts/<role>.md     each role's composed system prompt, snapshotted at every launch
├── plan.md               PlanDrafter (+ `## Open risks` if the cap was reached)
├── vocabulary.yaml       the typed step→catalog mapping and dispatch queue
├── surface.md            the surface survey, unabridged (preflight)
├── progress.yaml         every agent's typed decision, by stage
├── journal.md            the run's timeline, written as events happen
├── reports/NN-<role>-<what>.md   each dispatch's prompt and result
├── reports/premerge.md   the pre-merge pack, after a clean audit
├── decisions/<unit>-<role>-<turn>.yaml   typed decisions, when MCP tools are unavailable
├── certification.md      the serving envelope and certification plan (preflight)
├── early_checks.md       delivery checks run early, as warnings
├── evidence/             external matrix evidence, one YAML per owed cell
├── ledger.jsonl          every verdict and what it was measured on
├── gate_results.jsonl    per-prompt gate baselines and open findings
├── rerun_queue.json      stale or failing verdicts delivery will re-run
├── progress.json/.html   generated from the ledger, checkpoint and journal
├── progress.overlay.yaml the narrative those reports carry (yours to edit)
├── final-report.md       generated tables, narrative outside them
├── embargo/attempts.jsonl  every refused or redacted read of an embargoed path
├── overlap/<unit>/tree   the next unit's early worktree, under --overlap-units
├── boot.yaml             the boot gate's prompts and keywords (optional)
├── bin/commit.sh         the locking, self-verifying commit helper
├── .commit.lock          its lock file
├── .claude/skills/       this run's copy of the packaged skills
├── .agent-flow/          system prompts too long for argv, when there are any
├── dispatch/             turns, under --agent-transport external
├── TURN_INSTRUCTIONS.md  the driver's standing instructions, under --agent-transport external
├── archive/              the previous ledger and reports, after --clean
└── .trtllm_modeling_v2_state.json
```

The journal and the reports stay here rather than in the repository. They are
real evidence and worth keeping, but they are evidence *about a run*, and a
product tree is the wrong home for it — only a distilled record travels into
the repo at delivery.

`vocabulary.yaml` is the one artifact with a schema, because the orchestrator
walks it. `vocabulary.py` validates it and will reject a malformed one:
`OK`/`CERT` rows quote the cell, `note` line or `is_valid` assertion that
decides coverage, a queue `target` is `<category>/<name>` (shared catalog) or
`ops/<category>/<name>` (the target's own `ops/`), `GAP` rows name a candidate, `verify` dispatches name the surface, the two chapters do not
overlap, and a `ceiling` verdict carries its evidence.

A `GAP` row names its rung on the sourcing ladder, cheapest first:

| tier | source | the row names |
|---|---|---|
| 1 | a `torch.ops.trtllm.*` op | the op |
| 2 | an in-tree callable under `tensorrt_llm/_torch/modules/**` | the callable |
| 3 | FlashInfer's public API, for a kernel no trtllm op wraps | `flashinfer.<module>.<name>` |
| 4 | native torch, composed by this run | what tiers 1-3 offered (`tried`) |
| 5 | a Triton kernel, written by this run | what tiers 1-3 offered (`tried`) |

The op inventory sees tier 1 only. FlashInfer is a pinned dependency and the
tree wraps few of its kernels, so the surface survey also reads the installed
package's public API off disk and records which callables the engine already
references; `surface.md` lists them. A ceiling accounts for all five tiers.

A queue row may also name the `unit` that first needs its entry (`core` when
absent). Every row is dispatched in the smith phase regardless, but only a
failure core needs — by its tag, or as a transitive `depends_on` of a core row —
closes the run there (`smith-blocked`). Any other failure is recorded
`deferred` and retried once before its unit's first assembler turn; if the
retry fails too, the run closes `smith-blocked:<unit>`.

## Flags

| flag | default | |
|---|---|---|
| `--task` | required | the spec. Validated before any agent is constructed |
| `--workspace` | `workspace/trtllm-modeling-v2` | |
| `--dry-run` | off | preflight, then stop. Leaves the checkpoint at `preflight`, so the real run re-claims the devices |
| `--max-plan-rounds` | `1` | drafter, reviewer, at most one re-draft, then proceed regardless |
| `--max-assembler-turns-per-unit` | `6` | bounds the self-heal loop |
| `--max-parallel-smiths` | `10` (`8` with `--agent-transport external`) | where extra compute buys wall-clock. Each dispatch is a full agent session and the upstream rate limit is reported without backoff, so this is bounded by default; `0` opens the whole batch and on a typical queue that is every entry at once |
| `--reviewer` | `light` | `light` reads the evidence and runs nothing; `full` re-runs the boot gate; `off` skips the turn |
| `--plan` / `--vocabulary` | — | skip the plan phase with pre-written artifacts |
| `--wait-for-spool` MINUTES | 0 | with a `session_spool` environment, wait up to MINUTES for the spool to appear (its session is still importing the image) before validating the task |
| `--skip-substrate-gate` | off | skip the stock-path cold start in preflight. An escape hatch, not a speed-up — see the flag's own help |
| `--agent-transport` | `inprocess` | how a role turn is delivered. `external` writes it to `<workspace>/dispatch/` and waits — see below |
| `--agent-turn-timeout` | per role | seconds one externally-dispatched turn may take, for every role. Without it each role has its own bound: assembler 16 h, smith and auditor 6 h, plan drafter and reviewer 4 h, plan reviewer and runner 3 h |
| `--overlap-units` | off | start the next unit's assembler in its own worktree once the current unit is `UNIT_READY` — see below |
| `--stage` | — | `tune`: start at (or re-enter a delivered run at) the performance stage. Same as the `tune` subcommand |
| `--clean` | off | wipe the checkpoint and managed files |

Subcommands and tools that are not a run:

| command | does |
|---|---|
| `trtllm-modeling-v2 perf {ladder,campaign,table} ...` | perf tooling outside a run: drive the config ladder, inspect a campaign, print a curve table -- see [The tune stage](#the-tune-stage) |
| `trtllm-modeling-v2 premerge <step>` | the pre-merge pack — see [Commits and the pre-merge pack](#commits-and-the-pre-merge-pack) |
| `trtllm-modeling-v2 gate <name> --task task.yaml [--unit U] [--member M] [--tree T]` | one gate or witness (`boot`, `accuracy`, `acceptance`, `scheduler-equivalence`, `launch`, `witness`) run standalone, exactly as the orchestrator runs it |
| `trtllm-modeling-v2 session {render,status,requests,submission,allocate,ack,watchdog} ...` | compute sessions: entry scripts, status, hand-over requests — see [Execution substrate](#execution-substrate) |
| `trtllm-modeling-v2 driver --workspace W {wait,pending,status,claim,release,prompt,answer,instructions}` | the driver side of `--agent-transport external` |
| `python -m agent_flow.workflows.trtllm_modeling_v2.envelope {migrate,plan} task.yaml` | adopt a declared serving envelope; print the certification plan |
| `python -m agent_flow.workflows.trtllm_modeling_v2.report <ws> [--final-report] [--check DOC...]` | regenerate the reports, or check documents against the ledger |
| `python -m agent_flow.workflows.trtllm_modeling_v2.freshness {status,check-merge} <ws> --repo R` | stale verdicts and the re-run queue; whether a tree is the one regressed |
| `python -m agent_flow.workflows.trtllm_modeling_v2.ledger record <ws> ...` | record a verdict measured outside the workflow |
| `python -m agent_flow.workflows.trtllm_modeling_v2.smith_brief --workspace W <request>` | one smith's call-site brief, by hand |

### `--overlap-units`

Units stay ordered — core is green before the first feature — but a unit's
gates take hours during which the next unit's assembler has nothing to wait
for. With this flag, as soon as unit N's assembler writes `UNIT_READY`, unit
N+1's assembler starts in `<workspace>/overlap/<unit>/tree`: a git worktree
branched from `HEAD`, with the main checkout's untracked native build linked
in, its own `commit.sh`, its own progress file and decision path. When the
loop reaches N+1 (N's gates passed), the early result is adopted only if it
ended `UNIT_READY`, committed, left its worktree clean, rebases onto the
current `HEAD` without conflict and fast-forwards the main checkout; N+1 then
goes straight to its gates. **Anything else falls back to sequential**: the
main checkout is untouched, the worktree is kept with `NOT_ADOPTED.md` and
`early.patch`, and N+1's assembler is dispatched as usual. A worktree left by
a process that died is never adopted; remove it with
`git worktree remove --force`. Off by default (`overlap.py`).

## Running where the orchestrator cannot authenticate an agent

On a sandboxed host the spawned CLI cannot write its credential lock, and the
turn fails with `EROFS ... credentials.lock` reported as
`stop_reason='stop_sequence'` and **zero tokens** — shaped exactly like a model
refusal when in fact no agent started. Two things happen, in order:

1. **`agent_home.py` runs automatically.** If `$HOME` is not writable it stages
   one under the workspace, seeds it from the real credential store and exports
   `HOME` / `XDG_CONFIG_HOME` / `CLAUDE_CONFIG_DIR` before any agent exists.
   Nothing to configure, and an ordinary workstation is untouched.
2. **If that authenticates and then stops working**, the cause is rotation, not
   permissions: the staged store holds a copy of a refresh token the parent
   also holds. Copying harder does not fix a single-use token. Use
   `--agent-transport external`.

External transport moves the **turn** and nothing else. The orchestrator writes
`dispatch/<id>.system.md`, `dispatch/<id>.prompt.md` and
`dispatch/<id>.request.json`, then blocks until a driver writes
`dispatch/<id>.result.md` followed by `dispatch/<id>.done`. The driver can be a
human, another session or a service; it is trusted with nothing, because every
*decision* still arrives through `collect_decision_file` — validated against the
same closed enum the MCP tool schema would have — and every *verdict* still
comes from a gate the orchestrator ran itself.

The one thing a driver must get right: **the role's system prompt travels with
the turn**. A driver that delivers only the user prompt is running a different
agent, and its answer still looks plausible. The request also names the
backend and model the role is configured for (`agent`); a driver that answers
every turn on one model turns a cross-model plan review into a same-model one.

## Execution substrate

On a Slurm/pyxis cluster a run stands on one or more **compute sessions**:
allocations whose main process is the spool scheduler
(`substrate/scheduler.py`, started by `spool_watcher.sh`). Work is a script
renamed into `<spool>/pending/`; the receipt is `done/<name>.rc`, written last,
with the verdict beside it in `done/<name>.json`.

**Declare the sessions, render their entries.** `environment.sessions` in
`task.yaml` names each session's spool, role (`gates`, `work`, `orchestrator`,
`any`), node count, GPUs and time limit; `environment.orchestrator` names a
GPU-free session for the orchestrator. `trtllm-modeling-v2 session render --task
task.yaml --name a` writes the session's entry script from that block, so no
entry script is hand-edited; `session submission` prints the broker call that
starts it, `session status` shows every session.

**The scheduler** is the default. `MV2FLOW_SPOOL_WATCHER=legacy` falls back
to the old flock watcher (`scripts/spool_watcher_legacy.sh`), and the
orchestrator stays correct under it. The scheduler gives each job what its
header asks for
(`spool_submit.sh --gpus N|all --wall 2h --tree <checkout> <name> '<cmd>'`):

- one priority queue; whole-node jobs go before slot jobs, and a waiting
  multi-device job holds the devices it needs, so nothing is starved;
- device slots, exported as `CUDA_VISIBLE_DEVICES`;
- a wall limit counted from the job's start, a resident-memory limit, and an
  optional host-memory pressure floor;
- the allocation's MPI context (`PMI*`, `PMIX*`, `OMPI*`, `SLURM*`, ...) removed
  from every job, core dumps off, processes a finished job left behind reaped;
- names checked against the requester's prefix (each external turn gets one),
  and earlier results of a reused name archived rather than deleted;
- **the safe-kill rule**: it stops its own jobs (limits, `spoolctl.sh cancel`)
  but never signals a process that still carries the allocation's MPI context;
  such a job is marked failed and its devices are quarantined until it exits.

**`--tree <checkout>`** runs a job with that checkout as its working directory
and first `PYTHONPATH` entry. Every job carries a resolve hook that records
which checkout each process imported; a result whose ranks disagree, or that
did not import the declared tree, is refused (`reason=mixed_checkout`). Gates
declare `trtllm_repo_path` as their tree.

**Native rebuilds** go through `scripts/incremental_build.sh`: it announces the
rebuild on every session (their schedulers hold new GPU jobs against that
tree), gives every installed library a fresh inode so mapped copies are never
rewritten, builds, checks the import and retracts the announcement.

**No hang needs a human.** A gate's deadline counts from its start; a command
whose session stops heart-beating fails instead of waiting; a job that prints
a traceback and then goes silent is marked `stalled`; an engine waits for the
host memory and `/dev/shm` it declares (the VMM release lag) and is refused
with the reason if it never comes back; `session_watchdog.sh` returns the
moment a session is stale, ending without a successor, or quarantined.

**Hand-over.** A session with a `time_limit` writes
`<spool>/handover/request.json` `handover_lead` before its end and stops
admitting jobs that would not finish, so they wait in the spool for the
successor that watches the same spool; once the successor is up the old
scheduler retires. The successor is acquired by the driver (`session allocate`
prints the broker call) or by an `allocator.command`; the orchestrator never
calls the cluster's scheduler.

**Multi-node.** A session with `nodes: 2` runs one scheduler per node;
`scripts/mn/mn_mpirun.sh` launches one MPI job across them through the
schedulers, with no sshd or srun in the container (`scripts/mn/README.md`).

**The external driver.** With `--agent-transport external`, the driver session
runs `trtllm-modeling-v2 driver --workspace <ws> wait`, which returns `NEW <id>`
(within an in-flight cap of 8), `ORCHESTRATOR_EXITED`, `SESSION_REQUEST` or a
session alert; `claim`, `prompt <id>`, `answer` and `status` complete the loop,
and `instructions` renders the standing `TURN_INSTRUCTIONS.md` from `task.yaml`.
A relaunched orchestrator adopts every unanswered turn rather than asking again.

## Commits and the pre-merge pack

Agents commit only through `<workspace>/bin/commit.sh` (`committer.py`). It
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
TROUBLESHOOTING.md.

Delivery runs the same rules over the branch: every commit since the merge
base that touches the run's allowed paths must be signed off by its author,
with no co-author, symlink or placeholder (`git:commit-hygiene`), or the run
closes `delivery-failed` and names `premerge resign` as the fix.

What a delivered branch still owes before it is a PR is
`trtllm-modeling-v2 premerge <step>`; once the audit is clean, delivery writes
the cheap half of it to `reports/premerge.md` with the remaining commands:

| step | does |
|---|---|
| `lint` | pre-commit twice over the changed files (`PRE_COMMIT_HOME` made writable); commits the formatting only when pass 2 is clean, with an AST proof that it changed nothing but formatting |
| `ci` | one test-db line, run as CI runs it (the line as `--test-list`, CI's flags and cwd), recording `HEAD` at start and end |
| `copyright` | NVIDIA headers on new source files, current year on modified ones |
| `hygiene` | per commit: sign-off by the author, no co-author, no symlink, no placeholder |
| `split` | the branch split by path into PR stacks, with checks that they reproduce it |
| `resign` | a new branch with missing sign-offs added, each signed as the commit's own author |
| `pr-open` | opt-in: a draft PR through the GitHub API; a dry run unless `--yes`. Nothing in the pack pushes |

## Source embargo

`source_embargo` in `task.yaml` lists paths no role may read. Entries take
three forms:
- a repo-relative path, for an implementation of this exact checkpoint in the
  tree;
- an absolute path outside the repo, for a sibling engine's checkout (a
  directory embargoes its whole tree);
- `package:<path>`, inside this workflow package, for its own worked
  examples, e.g. `package:examples/<model>.md`.

On every in-process role whose backend runs SDK hooks (`embargo.py`, through
`BackendConfig.hooks`):
- a `PreToolUse` hook refuses `Read`/`Edit`/`Write` on those paths, `Grep`
  inside them, and shell commands that read them;
- a `PostToolUse` hook removes their lines from a broad search's output.

Running the other implementation, as an accuracy oracle, stays allowed.

Where no hook can run, the role's system prompt says so and restates the list
with absolute paths. That covers a role on `codex`, which refuses per-client
callbacks, and a turn on the external transport, which runs in a session the
orchestrator does not own.

Every refusal, redaction and instruction is logged to
`<workspace>/embargo/attempts.jsonl`.

## The tune stage

Onboarding makes a target *correct*; `tune` makes it *fast* without making it
less correct. Perf is measured, never a release gate -- but every kept change
passes the accuracy gate, and every number traces to a label dir.

```bash
trtllm-modeling-v2 tune --task task.yaml --workspace W        # standalone, or re-enter a delivered run
# chained: perf.enabled: true in task.yaml -> deliver, then tune, closes `tuned`
trtllm-modeling-v2 perf ladder run --spec tune.yaml           # tooling: the ladder alone
trtllm-modeling-v2 perf campaign status --spec tune.yaml      #          candidates and verdicts
trtllm-modeling-v2 perf table --baseline A A B C              #          a curve table of label dirs
```

Three layers, cheapest first (`tune/`):

| layer | who | what moves | judged by |
|---|---|---|---|
| **ground truth** | Python | baseline (`modeling_v2: require`), stock reference (`"off"`), identity repeats = the noise floor | -- |
| **config ladder** (`ladder.py`) | Python | runtime knobs and the target's own env switches, one rung per whole-node job, in parallel across sessions; winners stacked and re-measured | the keep rule |
| **campaign** (`campaign.py`, `stage.py`) | `tuner` turns | the forward: fusion, ordering, overlap, kernel source; one candidate = one worktree = one hypothesis | e2e A/B + accuracy gate, in Python |

- **The keep rule** (`records.compare`): a WIN is at least `max(min_gain,
  noise_factor x spread)` better on some headline metric (tok/s/user at the
  lowest concurrency, tok/s/GPU at the highest) and no worse than
  `max(1%, noise_factor x spread)` on any. The spread is measured, not
  assumed: identical-config repeats in the ground-truth phase.
- **The ladder is Python** for the reason preflight is: a rung needs no
  judgement once it is written down. Module knobs (`moe_config`,
  `attn_backend`, `use_cute_dsl_*`, `nvfp4_gemm_config`) are skipped for a
  target -- it pins its kernels through the catalog, so they never reach it.
  A task adds rungs for the target's own switches (`perf.ladder.extra_rungs`).
- **The tuner claims, the orchestrator measures.** A tuner turn implements
  one hypothesis in `W/tune/worktrees/<slug>` (branch `tune/<slug>`, native
  build linked in), proves it at the entry level (cells, GPU test) and in a
  micro-benchmark (`tune/microbench.py`, CUDA-graph and eager timings with
  per-layer weight rotation), and answers `CANDIDATE`. Python then measures
  the worktree end to end (`#spool: tree=`), applies the keep rule, runs
  `gate_accuracy.py` (which forces `modeling_v2: require` into the LLM-API
  yaml), and cherry-picks a keep onto `tune/best`.
- **The roadmap** comes from a first tuner turn (profile + code reading ->
  `W/tune/roadmap.yaml`), or from the operator. Candidates run
  `perf.campaign.max_parallel_tuners` at a time; the stage stops after
  `max_consecutive_misses` misses or `max_iterations` candidates. A spent
  roadmap with budget left is refreshed by a roadmap turn on the moved best
  tree (`roadmap-2.yaml`, ...; `max_roadmaps`, default 2), told what was tried.
- **Stacked evaluation and batches.** A candidate is measured cherry-picked
  onto the current best tree, against a control of that tree on the same host
  (hosts differ by more than most candidates move). Winners on one best tree
  wait for each other and are stacked: the compatible set with the largest
  summed gain (stacking simulated in the object store, no checkout) is
  measured once and gated once, and must hold every member's best single
  gain. A winner left out conflicts with the batch in the forward file: it is
  sent to a REBASE turn onto the batch commit while the batch measures, waits,
  and stacks on the new best when the batch is kept. The same holds for winners
  that conflict with a lone winner being gated, and a gate whose stacked tree
  another keep moved is stopped and the candidate re-stacked.
- **The keep gate** runs the target's own test suite and the modeling_v2
  contract tests on the gated tree before gsm8k (`gate_accuracy.py --tests`):
  candidates each pass their own op tests, a stack of them can still break a
  block test none of them ran.
- **Failures that are not the candidate's** -- a session that dies with its
  job, a full filesystem -- re-queue the measurement (once, twice for the
  substrate's own return codes) instead of reverting the candidate; the
  watcher and the campaign loop retry their own writes on ENOSPC/EDQUOT.
- **Kernel sourcing for perf** adds two tiers to the onboarding ladder:
  FlashInfer *as installed* (a newer FlashInfer's kernel is an upgrade
  candidate, not a source) and Triton / CuTe DSL / CUDA written by the run.
  When several sources serve one op, all are micro-benchmarked and the
  alternatives are recorded in `kernel_choices.json`.
- **Profiling** goes through the substrate: `bench.py --nsys`. The default
  is session mode (`nsys launch` + an external `nsys start/stop` window after
  `--nsys-delay`), because the in-process `cuProfilerStop` of TensorRT-LLM's
  recipe (`-c cudaProfilerApi --capture-range-end=stop --cuda-graph-trace node
  --gpu-metrics-devices=none`, window from `TLLM_PROFILE_START_STOP`;
  `--nsys-mode api`) never returned on this server. `tune/nsys_summary.py`
  attributes per step class (decode / mixed / prefill), placing CUDA-graph
  replays at their launch.

The `perf:` block (validated by `tune/stage.py:validate_perf_block`; excluded
from the evidence spec, so adding it never stales a delivered verdict):

```yaml
perf:
  enabled: true                     # chain after deliver
  identity_configs: [gate.yaml]     # LLM-API yaml(s): gate runtime settings + workload capacity
  workload: {isl: 8192, osl: 1024, concurrency: [1, 128]}
  reference: stock                  # stock = modeling_v2: "off", same configs; or none
  spools: [<whole-node perf sessions>]
  dev_spools: [<1-GPU sessions for tuner jobs>]
  env_script: <sourced by every perf job>
  noise_repeats: 2
  ladder: {min_gain: 0.01, noise_factor: 3.0, extra_rungs: [{name, overlay, why}]}
  campaign: {max_parallel_tuners: 4, max_iterations: 12, max_consecutive_misses: 3,
             accuracy: {task: gsm8k, anchor: 96.36}}
  source_embargo: [<the checkpoint's in-tree implementation, its kernels and ops>]
```

Skill: `skills/trtllm-modeling-v2-perf-tuning/` (campaign discipline, the candidate
protocol, kernel sourcing, graph safety; `references/nsys.md`,
`references/modeling-iterations.md`).

## What this workflow does not do

- **Performance as a release gate.** Perf is measured, never gated: a target
  whose gates are green is deliverable whether or not `tune` ran.
- **Isolation of the model zoo.** No blanket hook denies agents read access to
  TensorRT-LLM's per-model definitions: the zoo is an ordinary read. What carries the duty instead is the
  `modeling-core` invariants — a copied forward cannot pass the
  closed-vocabulary audit, is not flat, and leans on shared helpers a target
  may not have — plus the rule kept verbatim in all three writing roles:
  *read to understand an interface, never to copy a structure*.

  A **targeted** embargo is enforced instead: see
  [Source embargo](#source-embargo).

## Tests

```bash
cd agent-flow && pytest tests/workflows/trtllm_modeling_v2/
```

Every test runs on CPU with no model, no GPU and no cluster: the probes and
the shell are injected, and the agent layers are replaced by a fake that
writes the typed decision a real role would have written. A few earn their
keep by reading the real checkout, and each reads only what main has:

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
ships.

## Feature increments

`task.yaml`'s `features:` is a closed list, because a feature with no skill is
an assembler told to build something nobody wrote down how to build.

| feature | skill | gated by |
|---|---|---|
| `mtp` / `eagle3` / `dflash` / `dspark` | `trtllm-modeling-v2-speculator` | boot, accuracy, **acceptance** |
| `eplb` | `trtllm-modeling-v2-eplb` | boot, accuracy |
| `dwdp` | `trtllm-modeling-v2-dwdp` | boot, accuracy |
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
without the offload, token-identical output across the disaggregation hop, or
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
