# perf-optimize

Iteratively **applies** TensorRT-LLM serving optimizations — the acting
counterpart to `perf-analyze` (which only diagnoses). Measures a
`trtllm-serve` baseline, plans evidence-grounded optimizations into a
machine-readable `roadmap.yaml`, applies them one at a time, gates every
change on code quality / functionality / measured gain (with an nsys
capture of every accepted state), runs the configured number of rounds,
independently verifies the final state, and reports expected-vs-measured
results.

```
benchmarker ──▶  projector  ──▶ ┌── round loop (max_rounds; deterministic breaks) ──┐ ──▶ qa ──▶ reporter
 (baseline)     (SOL ceiling,   │ analyzer ──▶ ┌─ item loop (≤ max_items_per_round)─┐│   (final     (report)
                 on unless      │ (roadmap     │ optimizer ◀──▶ evaluator           ││    verification)
                 sol.enabled:   │  ranked by   │ (apply next    (gate → APPROVE |   ││
                 false)         │  benefit;    │  pending item)  REJECT | PUSH_BACK;││
                                │  profiles    │                 ≤ max_attempts     ││
                                │  after an    │                 _per_item retries) ││
                                │  accept,     └────────────────────────────────────┘│
                                │  else re-plans, no GPU)                             │
                                └─────────────────────────────────────────────────────┘

--reuse-analysis <dir> imports a previous perf-analyze / perf-optimize
run's baseline, SOL projection and profile, starting the campaign at the
optimize stage (round 1's analyzer then plans without profiling).
```

- **benchmarker** — serves the checkpoint, runs the canonical
  `benchmark_serving.py` at the configured operating point(s) (a
  `benchmark.concurrency` list means one run per point — Pareto-curve
  mode), writes `baseline/benchmark_results.md`. This anchors
  `roadmap.yaml`'s `baseline.value` (and `baseline.curve` in curve
  mode).
- **projector** *(on by default — skipped only when `task.yaml` sets
  `sol.enabled: false`)* — runs
  **once per campaign**, between the baseline and round 1: derives the
  analytical speed-of-light (SOL) ceiling for this model/hardware/
  operating point per the `internal-perf-sol-analysis` skill (hardware
  peaks from the skill's calculator, latency constants measured when a
  GPU is reachable, the α-β-u arithmetic written out, the model
  architecture read from the checkpoint's `config.json`), and writes
  `sol_projection.md` with a
  baseline-vs-SOL gap analysis (curve mode: per point), plus the
  machine-readable `sol_work/peaks.json` the analyzer's per-round
  correlation joins against. The ceiling is
  a property of the hardware + model + operating point — later rounds
  compare against the same projection rather than re-deriving it.
- **analyzer** — once per round (with `--reuse-analysis` round 1's
  profile is imported instead and the analyzer runs plan-only; after a
  round that neither accepted anything nor made a potentially
  build-changing code attempt it runs **replan-only**, planning from the
  standing profile and that round's verdicts — see *What a round costs*):
  profiles the *current* build (nsys +
  torch profiler + an ncu per-kernel deep dive on the top nsys kernels,
  captured over the same iteration window and interpreted with the
  `perf-nsight-compute-analysis` skill — per-kernel SOL%, occupancy,
  warp stalls → bound class; perf-analyze methodology), then
  writes/updates
  `roadmap.yaml` — items ordered by `expected_gain_pct` (bottleneck share
  removed, casebook-grounded), never by fix ease, each item's evidence
  drawn across the analyses (nsys timeline, ncu kernel analysis, SOL
  correlation when the projector ran) rather than the timeline alone.
  Later rounds re-profile
  (bottlenecks shift after each accepted item) and update statuses /
  ordering without rewriting history — a round following one that left
  the standing runtime profile current has no shift to find and re-plans
  instead. When the projector ran, it reads
  `sol_projection.md` as context (not evidence): the projected headroom
  and bound mix inform the ranking and sanity-bound each item's
  `expected_gain_pct`; measured trace evidence always outranks the
  projection. Each profiling round also runs the skill's **measured↔SOL
  correlation** (`sol_calc.py analyze`): the round's traces roll up
  into `analysis/regions.json`, join against the projector's
  `sol_work/peaks.json`, and the resulting per-op table (% of SOL,
  gap, bound per region) lands in the findings' *SOL correlation*
  section — the sharpest re-ranking signal the roadmap gets. And it
  never exhausts the roadmap silently: leaving no
  actionable pending item while projected headroom remains obliges a
  *Remaining-gap attribution* section in `profile_findings.md` — every
  part of the gap gets a new item or an evidence-backed reason it
  cannot be closed in this campaign (unexplained parts stay labeled
  unexplained).
- **optimizer** — applies pending roadmap items **one at a time**
  (top-1 first; up to `max_items_per_round` per round, so several items
  share one analyzer profile while each keeps its own evaluation):
  `approach: config` edits `tuning/extra_llm_api_options.yaml`;
  `approach: code` edits the TRT-LLM source (installed-package check
  first). Smoke-checks the server, never benchmarks, never commits.
  When the projector ran, it reads `sol_projection.md` as context (not
  spec): where the item leaves a choice of realization variants or knob
  values, it aims at the binding ceiling and records an `SOL alignment`
  line in its summary — the projection never expands the item.
- **evaluator** — reviews the diff, verifies functionality (sanity
  completions; targeted tests for code items), re-measures with the
  canonical benchmark, and applies the acceptance gate (below). Emits a
  structured three-way verdict with a reason category: **APPROVE**
  accepts the attempt, **PUSH_BACK** loops back to the optimizer with
  actionable feedback (up to `max_attempts_per_item` attempts total),
  and **REJECT** fails the item terminally — the judge's call that no
  retry would help, saving the benchmarks a doomed retry would burn.
  Stateless — every attempt is judged with fresh eyes. Negative
  verdicts close with a one-line *Gap implication*
  (mechanism-already-present / mechanism-inapplicable /
  applied-but-no-gain / blocked-by-constraint) — the projection-free
  evidence the analyzer's re-planning and the report's remaining-gap
  accountability are built from.
- **accept-evidence capture** — inside the evaluator's APPROVE turns,
  not a separate stage: after the clean measurement, the evaluator
  relaunches the server under the canonical nsys wrap, replays the load
  once, and saves `attempt_<k>/profile/` (trace, `nsys_stats.txt`,
  replay log), then writes a *Kernel evidence* section comparing the
  capture against the previous capture of the accepted state (the
  round's analyzer profile, or the prior accept) — verifying the item's
  claimed mechanism is actually visible in the trace. Because rejected
  attempts are hard-reverted, the **last accept's capture is always a
  profile of the final accepted state** — the reporter's "after" side.
  The capture is diagnostic, never a measurement (fresh relaunch; the
  verdict comes from the un-profiled run); no capture is made when
  `nsys` is not in `profile.methods`, and a failed capture never flips
  a verdict.
- **qa** — the campaign's **final verification**, run once after the
  round loop (and skipped when no item was accepted): stateless
  fresh-eyes benchmark + sanity completions (+ an accuracy eval iff
  `task.yaml` configures one), computing the verified cumulative
  improvement vs baseline that headlines the report. It makes no
  loop decision — the loop is already over.
- **reporter** — synthesizes `optimization_report.md` + a 1:1
  `optimization_report.html`: verified cumulative improvement, the
  optimization trajectory (baseline → each accepted item → final
  verification, rendered as a line chart in the HTML),
  expected-vs-measured per applied item, a kernel-level before/after
  comparison from the round profiles and accept-evidence captures,
  failed attempts with reasons, final config/code diff, and the
  remaining roadmap as future work. When the projector ran, the report
  gains a *Projection vs Measured* section (baseline vs final % of SOL
  — how much of the projected headroom the campaign captured; the
  curve-mode Pareto chart overlays the SOL-projected ceiling as a
  dotted third polyline), closed by a **remaining-gap accountability**
  breakdown: every part of the remaining gap-to-SOL is verdicted
  `closed` / `infeasible: <constraint>` / `untried` / `unexplained`,
  each verdict citing an artifact (a failed item's *Gap implication*,
  a round's *Remaining-gap attribution*, the projection's caveats) —
  a campaign may end short of the ceiling, but never without saying
  why.

The orchestrator — not the agents — owns the roadmap lifecycle fields and
the git state of the TRT-LLM checkout, driven by the evaluator's
structured decisions in `progress.yaml`.

## The acceptance gate

An attempt is **APPROVEd** only when all three hold:

1. code quality OK (scoped, clean diff), and
2. functionality OK (server serves coherent completions; targeted tests
   pass for code items), and
3. `measured_gain_pct ≥ accept_fraction × expected_gain_pct` **and**
   `measured_gain_pct ≥ noise_floor_pct`, measured on
   `optimize.target_metric` against the **last accepted** measurement
   (`current_best` in `roadmap.yaml`), so gains accumulate.

When any axis fails, the evaluator chooses between two negative
verdicts: **PUSH_BACK** (a concrete, actionable fix exists — the
optimizer retries with the feedback, bounded by
`max_attempts_per_item`; on the final attempt PUSH_BACK coerces to
REJECT) or **REJECT** (the item's premise is broken — no retry would
help; the item is failed immediately and the loop moves on). Either way
the orchestrator reverts every change.

**Pareto-curve mode** (`benchmark.concurrency` is a list): every
measurement runs once per concurrency point (over one server launch) and
gains are computed per point against the same-concurrency
`current_best.curve` entry. Rule 3 becomes the **Pareto gate**: the
**mean** per-point gain must pass both thresholds **and** no individual
point may regress by more than `noise_floor_pct` — a trade that helps
one regime by hurting another is rejected. The ledger
(`baseline`/`current_best`) then carries a `curve` of per-point
`{concurrency, value, tok_s_user, tok_s_gpu}` rows (tok/s/user =
`1000/mean_tpot_ms`, tok/s/gpu = `output_throughput/num_gpus`), and the
report gains a *Pareto Improvement* section + chart (x = tok/s/user,
y = tok/s/gpu, baseline vs final). Note the benchmark cost multiplies by
the point count on **every** measurement (baseline, each evaluator
attempt, the final verification) — keep the list short (~3–5 points).

## When the loop stops

No agent decides when to stop. The loop runs exactly
`optimize.max_rounds` rounds unless one of two deterministic,
orchestrator-enforced breaks fires first:

- **Roadmap exhausted on an unchanged build** — no pending item
  promises at least `noise_floor_pct` through an allowed approach
  (checked after every analyzer turn and after every item's terminal
  outcome) *and* nothing has been accepted since the analysis that
  planned it. A fresh profile would then find the same nothing, at full
  profile cost. When the roadmap runs dry with accepts outstanding the
  loop does **not** close: the build those accepts produced has never
  been analyzed, so it spends one more round profiling what they exposed
  (budget permitting) and closes on *that* verdict.
- **Target met** — the optional `optimize.target_improvement_pct` is
  reached by the roadmap ledger's cumulative gain (`current_best` vs
  `baseline`; curve mode: the mean of per-point gains), checked after
  every accepted item.

Either way the campaign proceeds to the one-shot final verification
(skipped when nothing was accepted) and the reporter.

## What a round costs: profile or replan

Every round opens with an analyzer turn, but not every round pays for a
profile. Rejected config attempts are hard-reverted (`git reset --hard`
plus the last accepted tuning config), so they leave the runtime the
standing analysis describes. Code attempts are different: `clean -x` is
deliberately omitted, so a rebuilt gitignored `.so` or JIT/AOT cache may
survive the source revert. The orchestrator records that uncertainty and
profiles rather than pretending the old traces are current.

- **Profiling round** — round 1; any round opening after an accept; and
  any round whose reverted code attempt may have changed ignored build
  output. Re-profiles the current runtime (nsys + torch + ncu per
  `profile.methods`) and re-ranks the roadmap against fresh traces. An
  older checkpoint with no profile-currency marker also buys one
  conservative profile on resume.
- **Replan-only round** — opens when the standing profile is known to be
  current: the predecessor accepted nothing and made no code attempt
  capable of leaving rebuilt output behind. The analyzer launches no
  server and runs no profiler; it plans from the standing analysis plus
  the round's evaluator verdicts, marking
  disproven items obsolete, bounding the gains the measurements cap, and
  adding what the failures imply. Those verdicts are the round's real
  yield — an item measured dead is evidence about *this* build — and
  converting them into roadmap edits is what the turn is for. The
  per-kernel ledger contract is waived for such a round (it ran no ncu);
  the standing ledger still describes the build.

Neither mode is an agent's choice, and a replan round is not a skipped
round: if it leaves nothing actionable, that is the roadmap-exhausted
break and the campaign closes.

That break only fires at the top of a round, never mid-round. A roadmap
that runs dry between items ran dry against a plan written *before* the
round's measurements existed, so the loop spends one more round — free
when nothing was accepted, a profile when something was — and closes on
the plan the analyzer makes against them. What ends a campaign is an
analyzer turn that has seen the evidence and still finds nothing, not
the plan simply running out.

## Reusing a previous run's analysis

`--reuse-analysis <dir>` seeds a fresh workspace from a previous
`perf-analyze` run or `perf-optimize` campaign, so the campaign starts at
the optimize stage instead of re-deriving what that run already measured:

| imported | from a perf-analyze workspace | from a perf-optimize workspace | replaces |
| --- | --- | --- | --- |
| baseline report + result JSONs | `benchmark_results.md` | `baseline/benchmark_results.md` | the benchmarker |
| SOL projection + `sol_work/` | `sol_projection.md` | `sol_projection.md` | the projector |
| profile findings + traces (+ `kernel_ledger.yaml`) | `profile_findings.md` | newest `rounds/round_<n>/analysis/` | round 1's profiling |
| roadmap (as read-only prior art) | — | `roadmap.yaml` | nothing |

Round 1's analyzer then runs **plan-only**: it reads the imported
evidence, checks that it actually describes this task (same model,
parallel mapping, operating point), runs the dormant-capability sweep,
and writes `roadmap.yaml` — launching no server, no profiler, and no
benchmark. Round 2 profiles normally: the imported traces describe
*another* run's build, so they never stand in for one this campaign
made, and the replan rule only ever plans from a profile of this
campaign's own checkout.

Two deliberate limits:

- **Ledger state is never imported.** A source `roadmap.yaml` lands in
  `reused_analysis/prior_roadmap.yaml` as reference material only — its
  statuses, `measured_gain_pct` and `current_best` describe the *source*
  campaign's checkout, not this one's. The plan-only analyzer weighs it
  as evidence (carry the pending items forward, don't re-propose what
  failed) rather than inheriting it.
- **The baseline is inherited, not re-measured.** Every gain this
  campaign reports is computed against numbers measured by the source
  run, so the two must describe the same system. The import writes
  `reused_analysis/manifest.md` recording what came from where, the
  analyzer owes a fit check against it, and the report says so in
  Configuration. If the hardware or checkpoint differs, don't reuse the
  baseline — the import is per-artifact, so a source without one simply
  gets benchmarked normally.

Fresh runs only: on resume the checkpoint wins and the flag is ignored
with a warning. Note that a run whose `profile.kernel_coverage` contract
is on does **not** enforce the per-kernel ledger for a reused round that
carries none, nor for a replan-only round (neither ran ncu); every round
the analyzer actually profiles is still bound by it.

## Usage

```bash
# from the repo root
pip install -e .

perf-optimize --task path/to/task.yaml --workspace workspace/perf-optimize/my-model

# resume after a crash / Ctrl-C: just re-run the same command.
# start over from scratch:
perf-optimize --task path/to/task.yaml --workspace workspace/perf-optimize/my-model --clean
# override the round budget on a fresh run:
perf-optimize --task ... --workspace ... --max-rounds 5
# start at the optimize stage, reusing a previous run's analysis:
perf-optimize --task ... --workspace ... --reuse-analysis workspace/perf-analyze/my-model
```

## task.yaml

See `task.example.yaml`. Fields beyond the perf-analyze base spec
(`checkpoint_path`, `trtllm_repo_path`, optional `extra_llm_api_options`
/ `benchmark` / `profile` / `slurm-environment` / `sol` — see the
perf-analyze README for the base fields, including
`benchmark.concurrency`, an int or
a list of ints, and `benchmark.num_prompts`, an int or — curve mode
only — a list paired index-by-index with the concurrency list so
low-concurrency points can run fewer prompts; the pre-rename
`max_concurrency` spelling fails validation with a pointer to the new
name. The optional `sol` block — every field optional, `enabled` the
stage gate (default `true`) and `gpu` the part-name hint for the SOL
skill's peaks calculator — gates the one-shot SOL projector stage here
exactly as in perf-analyze):

| field | required | default | meaning |
| --- | --- | --- | --- |
| `optimize.max_rounds` | no | `5` | The number of rounds the loop **runs** (not just a cap — only the two deterministic breaks above end it earlier); each round is one analyzer turn + up to `max_items_per_round` items, so `max_rounds × max_items_per_round` bounds total items attempted. Only rounds with a stale or unproven runtime profile pay to refresh it (see *What a round costs*), so this bounds items far more tightly than GPU hours. |
| `optimize.max_items_per_round` | no | `3` | Items applied per round, **one at a time** — each still gets its own optimizer ⇄ evaluator gate, measured gain, and revert; the budget only sets how many share one analyzer turn. `1` reproduces the original one-item-per-round loop. |
| `optimize.max_attempts_per_item` | no | `3` | Total optimizer attempts per item: PUSH_BACK verdicts retry until this bound, then the item is marked `failed` and reverted (an explicit REJECT fails it immediately). |
| `optimize.approaches` | no | `[config, code]` | Which optimization approaches the run may plan/apply: `config` edits the live tuning YAML, `code` edits the TRT-LLM source. Restrict to `[code]` for a code-only campaign (no knob tuning) or `[config]` to leave the checkout untouched. Enforced in three layers: the analyzer only plans allowed items, the orchestrator never dispatches a disallowed pending item, and any attempt that edits through a disallowed approach (tuning file differs from the accepted snapshot / dirty worktree) is auto-rejected before the evaluator benchmarks it. |
| `optimize.accept_fraction` | no | `0.5` | Fraction of an item's `expected_gain_pct` the measured gain must reach. |
| `optimize.noise_floor_pct` | no | `1.0` | Minimum measured gain (%); also the actionability floor for pending items. |
| `optimize.target_metric` | no | `output_throughput` | Result-JSON key gains are computed on (see below). |
| `optimize.target_improvement_pct` | no | — | Optional early-stop: the orchestrator concludes the loop once the roadmap ledger's cumulative improvement reaches this. |
| `accuracy.command` | with `accuracy` | — | Accuracy eval command the final verification runs verbatim against the live server (e.g. `trtllm-eval ...`). Omit the whole block to skip accuracy checks. |
| `accuracy.baseline_score` | no | — | Reference score to compare against. |
| `accuracy.max_drop_pct` | no | `1.0` | Allowed relative score drop vs `baseline_score`. |

**Target metric keys** (from the `benchmark_serving.py` result JSON):
`output_throughput` (tok/s, default), `total_token_throughput`,
`request_throughput`, and latency keys `mean_ttft_ms` / `median_ttft_ms`
/ `p99_ttft_ms` (likewise `*_tpot_ms`, `*_itl_ms`, `*_e2el_ms`). Gains
are always normalized so positive = improvement (throughput up, latency
down).

## Git requirements (read before running)

`perf-optimize` **mutates the TRT-LLM checkout** at `trtllm_repo_path`:

- The checkout must be a git repo. Rejected attempts are reverted with
  `git reset --hard` + `git clean -fd` (without `-x`, so gitignored
  build artifacts survive) — anything uncommitted and unignored when
  the run starts is destroyed along with them, so commit or stash
  changes you care about before running.
- Work happens on a dedicated branch `perf-optimize/<workspace>-<ts>`
  created from the current HEAD; each accepted item becomes **one
  commit**; pushed-back and rejected attempts are reverted before the
  next attempt/item starts. `--clean` never touches the checkout —
  abandoned branches are left for inspection.
- `approach: code` items only take effect when the checkout is the
  installed package (editable install). The analyzer/optimizer verify
  `python -c "import tensorrt_llm; ..."` resolves into the checkout and
  fall back to config-only optimization when it does not. Note the
  interaction with `optimize.approaches: [code]`: a non-editable install
  then leaves the run nothing to do — use an editable install for
  code-only campaigns.
- `optimize.approaches` only restricts what the *loop* may change; the
  task's `extra_llm_api_options` seed still applies as the baseline
  config in every mode.

## Workspace layout

```
<workspace>/
├── task.yaml                        # resolved spec (defaults filled in)
├── roadmap.yaml                     # the ranked plan; statuses/gains updated as the loop runs
├── sol_projection.md                # projector's SOL ceiling + baseline-vs-SOL gap (blank when sol.enabled: false)
├── sol_work/peaks.json              # projector's machine-readable peaks (analyzer's correlation joins against it)
├── baseline/
│   ├── benchmark_results.md         # benchmarker's baseline report
│   └── serve.log, *.json            # baseline run artifacts
├── tuning/
│   ├── extra_llm_api_options.yaml           # live server tuning (optimizer edits this)
│   └── extra_llm_api_options.accepted.yaml  # last accepted snapshot (orchestrator-managed)
├── reused_analysis/                 # --reuse-analysis only
│   ├── manifest.md                  #   what was imported, and from where
│   └── prior_roadmap.yaml           #   source campaign's roadmap — read-only prior art
├── rounds/round_<n>/
│   ├── analysis/                    # analyzer: profile_findings.md, nsys/torch/ncu traces (+ regions.json / sol.json when the projector ran;
│   │                                #   + kernel_ledger.yaml with a profile.kernel_coverage block)
│   └── item_<j>_<id>/attempt_<k>/   # per item: optimization_summary.md, evaluation.md, result *.json
│       └── profile/                 # accept-evidence nsys capture (APPROVEd attempts only)
├── final_verification/
│   └── verification_report.md       # QA's one-shot independent verification (+ its artifacts)
├── optimization_report.md           # reporter deliverable
├── optimization_report.html         # self-contained interactive companion (1:1)
├── progress.yaml                    # structured audit log (agents write via MCP tools)
└── .perf_optimize_state.json        # resume checkpoint
```

## Notes

- **Agent operator guide.** Upstream agent-flow additionally ships a
  `perf-optimize` project skill that teaches Claude Code to reach for this
  workflow instead of hand-rolling a tune loop. It is not vendored here
  because it hardcodes one site's cluster and registry; drive the
  `perf-optimize` CLI directly, using this README and
  [`task.example.yaml`](task.example.yaml) as the operator guide.
- **Session scoping.** Agent sessions match each role's unit of work:
  the analyzer keeps one session across the whole campaign (it must
  remember the roadmap it authored), the optimizer's session spans one
  item's retry attempts and is reset between items, and the evaluator /
  qa run stateless — the judges always get fresh eyes, and no role drags
  a long campaign's stale context into later decisions. (The
  benchmarker, projector, qa, and reporter run once each.)
- **Serve tuning lives in the workspace.** Every `trtllm-serve` launch in
  this workflow passes
  `--extra_llm_api_options <workspace>/tuning/extra_llm_api_options.yaml`
  (seeded from the task's `extra_llm_api_options` file, else `{}`), so
  config optimizations are applied by editing that one file. The
  perf-analyze convention (flag only when task.yaml sets it) does not
  apply here.
- **Canonical command templates.** All measurements reuse perf-analyze's
  canonical `benchmark_serving.py` / `nsys profile` / `ncu` templates at
  the configured operating point(s) — one run per `benchmark.concurrency`
  point in Pareto-curve mode — so numbers stay comparable across the
  whole campaign. The analyzer's ncu deep dive is bounded
  (`--launch-count`, kernel filter from the top nsys kernels) and
  interpreted with the `perf-nsight-compute-analysis` skill; it degrades
  gracefully when `ncu` or the skill is unavailable.
- **Per-kernel coverage contract (optional).** A
  `profile.kernel_coverage` block in `task.yaml` (empty mapping =
  defaults: `min_share_pct: 0.5`, `coverage_target_pct: 95`) upgrades
  the ncu dive from "top nsys kernels" to **every kernel above the
  share bar** (enumerated from the fresh kern_sum, extended until the
  coverage target is reached, captured over up to 3 bounded ncu passes
  that re-filter on still-missing stems so once-per-step kernels are
  not starved by per-layer hot ones). Each round the analyzer must then
  answer two questions per enumerated kernel — *can it be made faster?*
  *can it be fused with its neighbors?* — in
  `rounds/round_<n>/analysis/kernel_ledger.yaml`: every row carries the
  kernel's ncu SOL metrics/bound class plus a `faster` and a `fusion`
  disposition, each either a roadmap item id or an evidence-backed
  dismissal (`at-sol-floor`, `below-materiality` with arithmetic,
  `multi-consumer-pinned`, `already-fused`, `phase-boundary`,
  `needs-rebuild`, ...); `needs-rebuild` is valid only when a
  written-from-scratch replacement kernel routed from the Python call
  site is also ruled out, not merely because the incumbent ships
  compiled; fusion rows record the observed neighbors from the trace. The orchestrator schema-validates the ledger after every
  analyzer turn (both dispositions per row, `item` refs resolving to
  real roadmap ids, coverage ≥ target) and **aborts the stage on an
  incomplete ledger**, so the campaign cannot conclude while a hot
  kernel's optimization or fusion possibility was never considered; the
  reporter's *Kernel Coverage* section resolves the final ledger's
  dispositions to campaign outcomes and itemizes the untried tail.
  Requires `nsys` + `ncu` in `profile.methods`; costs extra profiling
  wall-clock per round.
- **Optimization casebook.** The benchmarker/analyzer load the
  `trtllm-agent-toolkit:perf-optimization-casebook` skill as read-only
  reference; the optimizer uses it *actionably* (how-to-apply /
  verification / rollback guidance). All roles degrade gracefully when
  the skill is not installed.
- **SOL projection (default-on stage).** The projector runs unless
  `task.yaml` sets `sol.enabled: false`; it follows the
  `trtllm-agent-toolkit:internal-perf-sol-analysis` skill as its
  methodology. That skill is `internal-` prefixed, so open-source
  toolkit builds strip it while keeping `perf-analysis`; which of the two
  this session has is resolved **in Python** before the campaign starts
  (`perf_analyze.sol_methodology`, one ~1 s probe — a session
  connection, no model call — failing open to the SOL skill if it cannot
  run), so the projector is told to load a skill that is actually there.
  Without the SOL skill it loads `perf-analysis` instead and works the
  same methodology without a calculator: the peaks come from named
  sources, marked as not calculator-resolved, and no
  `sol_work/peaks.json` is written — so the analyzer's per-round
  correlation degrades to its honest "Correlation unavailable" line. It
  degrades to a "Projection unavailable" file when no ceiling can be
  grounded at all, and never fabricates one. It
  runs once per campaign: the ceiling depends only on the hardware +
  model + operating point, so every later round compares against the
  same `sol_projection.md`. Consumers: the analyzer (roadmap ranking
  context; each round it also joins its fresh per-op measurements
  against the projector's `sol_work/peaks.json` with the skill's
  `sol_calc.py analyze` and reports the joined table in
  `profile_findings.md`'s *SOL correlation* section; plus the
  remaining-gap attribution owed whenever it leaves
  the roadmap exhausted with headroom remaining), the optimizer
  (aiming each item's realization at the binding ceiling — context,
  never an expansion of the item), and the reporter (the *Projection
  vs Measured* headroom-captured section with its remaining-gap
  accountability breakdown); the evaluator and qa deliberately see
  nothing of it — their gates stay measured-vs-measured so an
  analytical model can never anchor a verdict. For a spec or mapping
  that stays uncertain, the projector is pointed at the
  `internal-glean-search` skill / `internal-glean-specialist` subagent
  as read-only reference, used only if it is installed in the
  session.
- **Cost.** The analyzer re-profiles every round that follows an accept
  or a potentially build-changing reverted code attempt (nsys + torch +
  the bounded ncu deep dive by default); set `profile.methods: [nsys]`
  to trim it. When the standing runtime profile is still current, the
  next round opens replan-only and pays no GPU time at all — see *What a
  round costs* above. Each evaluator attempt runs a
  full benchmark, each **accepted** attempt additionally pays one
  profiled replay (the accept-evidence capture), and the final
  verification runs one more benchmark at campaign end. The per-item
  evaluator benchmark is the irreducible price of per-item attribution;
  raising `max_items_per_round` amortizes the analyzer profile across
  more items, at the cost of applying later items in the round against
  a ranking profiled before the earlier ones landed. Under fixed-round
  semantics `max_rounds` is the primary cost knob — the loop will spend
  the whole budget unless the roadmap runs dry on an unchanged build or
  the target is met.
- **Local vs Slurm.** With a `slurm-environment` block, every
  server-launching role (all but the projector and the reporter) is
  augmented with the Slurm container-bootstrap guidance, exactly like
  perf-analyze. The projector launches no servers; under Slurm it runs
  on the login node and records the latency constants as unmeasured.
