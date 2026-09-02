"""Shared prose blocks for the perf-optimize role prompts.

The serve / benchmark / profiling command knowledge is imported from
``perf_analyze.prompts._common`` (the settled cross-workflow reuse
direction — ``modeling_bringup`` reuses ``agent_team`` the same way), so
the canonical templates stay defined in exactly one place — including
the projector's SOL methodology blocks (``SOL_PROJECTOR_*``), the
analyzer's findings contract (``PROFILE_FINDINGS_CONTRACT``), and the
measured↔SOL correlation recipe (``SOL_CORRELATION_METHOD``): this
workflow's analyzer is perf-analyze's analyzer plus the roadmap
machinery, and both compose the same fragments.
This module adds the blocks specific to the optimization loop: the
``roadmap.yaml`` contract, git discipline for the mutable TRT-LLM
checkout, the kernel-reuse rule, the evaluator's acceptance gate, the
measurement protocol, the live tuning config, the *actionable* casebook
variant for the optimizer, the optimization report's HTML companion
spec, and the SOL-projection consumption blocks (``SOL_ANALYZER_CONTEXT``
/ ``SOL_OPTIMIZER_CONTEXT`` / ``SOL_OPTIMIZE_REPORTER_GUIDANCE``,
appended only when the projector stage is enabled — the default, unless
``task.yaml`` sets ``sol.enabled: false``).
"""

from typing import Sequence

from agent_flow.workflows.perf_analyze.prompts._common import (
    BENCHMARK_FLAGS_REFERENCE,
    BOTTLENECK_TAXONOMY,
    CASEBOOK_CONSULTATION,
    DERIVED_METRICS_REFERENCE,
    EVIDENCE_DISCIPLINE,
    EXECUTION_SLURM_BOOTSTRAP,
    PROFILE_FINDINGS_CONTRACT,
    PROFILING_KNOB_VERIFICATION,
    PROFILING_RUNS_REFERENCE,
    SERVE_FLAGS_REFERENCE,
    SERVER_LIFECYCLE,
    SOL_CORRELATION_METHOD,
    SOL_METHODOLOGY_FALLBACK,
    SOL_PROJECTOR_INTERNAL_KNOWLEDGE,
    SOL_PROJECTOR_METHODOLOGY,
)
from agent_flow.workflows.perf_optimize.roadmap_schema import APPROACHES

__all__ = [
    "BENCHMARK_FLAGS_REFERENCE",
    "BOTTLENECK_TAXONOMY",
    "CASEBOOK_APPLY",
    "CASEBOOK_CONSULTATION",
    "DERIVED_METRICS_REFERENCE",
    "DORMANT_CAPABILITY_SWEEP",
    "EVIDENCE_DISCIPLINE",
    "EXECUTION_SLURM_BOOTSTRAP",
    "EXPECTATION_GATE",
    "GIT_DISCIPLINE",
    "KERNEL_COVERAGE_REPORTER_GUIDANCE",
    "KERNEL_REUSE",
    "MEASUREMENT_PROTOCOL",
    "OPTIMIZE_HTML_COMPANION",
    "PROFILE_FINDINGS_CONTRACT",
    "PROFILING_KNOB_VERIFICATION",
    "PROFILING_RUNS_REFERENCE",
    "ROADMAP_SPEC",
    "SERVE_FLAGS_REFERENCE",
    "SERVER_LIFECYCLE",
    "SOL_ANALYZER_CONTEXT",
    "SOL_CORRELATION_METHOD",
    "SOL_METHODOLOGY_FALLBACK",
    "SOL_OPTIMIZER_CONTEXT",
    "SOL_OPTIMIZE_REPORTER_GUIDANCE",
    "SOL_PROJECTOR_INTERNAL_KNOWLEDGE",
    "SOL_PROJECTOR_METHODOLOGY",
    "TUNING_CONFIG_NOTE",
    "approach_restriction_note",
    "kernel_coverage_analyzer_note",
]


# --------------------------------------------------------------------------- #
# The roadmap contract (analyzer / optimizer / evaluator / qa / reporter)
# --------------------------------------------------------------------------- #

ROADMAP_SPEC = """\
## The roadmap contract (`roadmap.yaml`)

`roadmap.yaml` is the machine-readable optimization plan the whole loop
runs on. Its exact shape:

```yaml
version: 1
target_metric: output_throughput      # key in the benchmark_serving result JSON
baseline:                             # analyzer writes this once in round 1; frozen afterward
  value: 1234.5                       # curve mode: the MEAN of curve[].value
  source: baseline/benchmark_results.md
  curve:                              # curve mode only (benchmark.concurrency is a list);
    - {concurrency: 8, value: 812.0, tok_s_user: 21.4, tok_s_gpu: 101.5}   # ascending,
    - {concurrency: 32, value: 1657.0, tok_s_user: 12.9, tok_s_gpu: 207.1} # one entry per point
current_best:                         # the last ACCEPTED measurement; seeded equal to baseline
  value: 1298.7                       # curve mode: mean across points; carries curve too
  source: rounds/round_1/attempt_1/evaluation.md
items:                                # pending items ordered by expected benefit, descending
  - id: opt-001                       # stable slug; never renumbered or reused across rounds
    title: Enable CUDA graphs for decode
    category: launch-host             # compute | memory-bw | kv-capacity | launch-host | communication
    approach: config                  # config (tuning YAML only) | code (source edit)
    evidence:                         # >= 1 entry; cite trace files with numbers
      - "nsys: 31% GPU idle from per-launch gaps (rounds/round_1/analysis/nsys_stats.txt)"
    casebook_ref: "launch storm at decode -> cuda-graph capture"   # optional; the matched casebook row
    expected_gain_pct: 12.0
    expected_gain_rationale: "idle share x casebook-typical recovery for this pattern"
    how_to_apply: |
      Add cuda_graph_config to tuning/extra_llm_api_options.yaml; no source edit.
    status: pending                   # pending | in_progress | accepted | failed | obsolete
    attempts: 0
    measured_gain_pct: null           # filled from the evaluator's measurement
```

Rules that keep the loop deterministic:

- **Enum fields are exact.** `category` is exactly one of `compute` |
  `memory-bw` | `kv-capacity` | `launch-host` | `communication` — never
  shorthand (`memory` is not a category; memory-bandwidth pressure is
  `memory-bw`). `approach` is `config` or `code`, `status` one of the
  five listed values. The orchestrator schema-validates the file the
  moment your turn ends and an invalid enum aborts the run.
- **List order is priority order.** Pending items are sorted by
  `expected_gain_pct`, descending — the orchestrator always picks the
  *first* `pending` item, so a mis-ordered list optimizes the wrong thing.
- **Expected gains are grounded, not vibes.** Every item cites the
  profiling evidence (trace file + numbers) and, when one matches, the
  casebook precedent its estimate leans on.
- **Ownership.** Only the **analyzer** writes item content (ids, titles,
  categories, evidence, gains, ordering) and may mark still-pending items
  `obsolete` when fresh evidence — a re-profile, or the verdicts a
  replan-only round plans from — shows they no longer apply. The
  **orchestrator** owns every lifecycle field: it flips `status` to
  `in_progress` / `accepted` / `failed`, counts `attempts`, fills
  `measured_gain_pct`, and advances `current_best`. No agent ever flips an
  item to `accepted`/`failed`, edits `attempts` / `measured_gain_pct` /
  `current_best`, or rewrites the history of already accepted/failed items.
- **Curve mode** (`benchmark.concurrency` in `task.yaml` is a list): the
  analyzer's round-1 authoring writes `baseline.curve` — one
  `{concurrency, value, tok_s_user, tok_s_gpu}` entry per configured
  point, ascending, from the baseline report's curve summary table — and
  seeds `current_best` equal to it (curve included). `value` is the mean
  of the per-point `value`s. The orchestrator advances `current_best`
  (including its curve) from the evaluator's structured progress entry;
  per-item `measured_gain_pct` is the **mean of per-point gains**. Scalar
  runs omit `curve` entirely.
- **Focus scoring** (`optimize.focus_concurrencies` in `task.yaml`, curve
  mode only, optional): when set, every scalar derived from the curve —
  `baseline.value`, `current_best.value`, per-item `measured_gain_pct` —
  is the mean over **only those points** (the campaign's scored regime),
  while every `curve` list still carries **all** configured points and
  regressions are still checked at every point. Expected gains and their
  rationales are then estimates on the focus regime, and items whose
  benefit lives outside it are mis-planned.
"""


# --------------------------------------------------------------------------- #
# Git discipline (optimizer / evaluator)
# --------------------------------------------------------------------------- #

GIT_DISCIPLINE = """\
## Git discipline (the TRT-LLM checkout)

This run works on a **dedicated optimization branch** in
`trtllm_repo_path` (the branch name is given in your instructions). The
orchestrator owns all git state — it commits each accepted item and
reverts rejected attempts with `git reset --hard` + `git clean -fd`:

- **Never run `git commit`, `git reset`, `git checkout`/`git switch`,
  `git stash`, or `git push`.** A commit or branch switch you make
  yourself corrupts the orchestrator's accept/revert bookkeeping.
  Read-only git (`git diff`, `git status`, `git log`) is fine and is how
  the change under review is defined.
- Keep the worktree containing **only the current roadmap item's
  changes**. Do not drive-by refactor, reformat, or fix unrelated code —
  the whole worktree is committed on accept and wiped on reject.
- **The code must stand on its own in the TRT-LLM repo.** An accepted
  diff outlives this run and is read by people who have never seen this
  workspace. Never write comments, docstrings, or names that reference
  the run's internals — roadmap item ids (`opt-008`), rounds/attempts,
  the perf workspace or its files, benchmark results, or this workflow
  (e.g. "See the opt-008 bench in the perf workspace" is meaningless to
  every future reader). Comment only what the surrounding code cannot
  say itself — the non-obvious constraint, the why of a chosen value —
  in the repo's own terms; the provenance story belongs in
  `optimization_summary.md`, not in the source.
- **Before ANY source edit, verify the checkout is the installed
  package** (verify before asserting):
  ```bash
  python -c "import tensorrt_llm, os; print(os.path.realpath(tensorrt_llm.__file__))"
  ```
  The printed path must resolve **under `trtllm_repo_path`** (an editable
  install). On a mismatch, source edits will not take effect in the
  served process — record that as a blocker in your output file and do
  not pretend the change was applied.
- The review basis for an attempt is
  `git -C <trtllm_repo_path> diff` (plus `--stat`) — uncommitted changes
  on the optimization branch. New files the attempt added show up with
  `git status --porcelain`; list them explicitly in your output.
"""


# --------------------------------------------------------------------------- #
# Kernel reuse (analyzer / optimizer / evaluator)
# --------------------------------------------------------------------------- #

KERNEL_REUSE = """\
## Prefer existing kernels over writing new ones

When an optimization calls for a kernel — a fusion, an attention or GEMM
variant, a norm/RoPE/quantization pattern — the fix is usually **wiring
up a kernel that already exists**, not authoring one. A hand-written
kernel that duplicates what an existing provider ships is the wrong
change whatever it measures: it forfeits the provider's tuning and
testing and adds unmaintained surface to the diff. The preference is
conditional on a suitable kernel existing, though — when the search
below comes up empty, **writing a new kernel is the encouraged
realization of the item, not a rule violation**: a real opportunity
abandoned as "no existing kernel fits" is a worse outcome than a scoped,
tested new kernel. Before any new kernel, search in priority order:

1. **The TRT-LLM checkout itself** (`trtllm_repo_path`) — the custom-op
   layer and kernel library already cover many fusions behind an op, a
   backend selector, or a config knob that this code path simply is not
   calling yet.
2. **flashinfer** — attention variants, fused norm + quantization, RoPE,
   sampling, and more; check the version installed in this environment
   and the call sites TRT-LLM already has for it.
3. **Any other kernel provider already integrated** in the checkout
   (CUTLASS / cuBLAS / cuDNN, DeepGEMM, vendored Triton ops, ...).

What this means per role:

- **Analyzer** — a roadmap item that needs kernel work must name, in
  `how_to_apply`, the existing kernel/op to wire up and where you found
  it. When your search comes up empty, still plan the item: have
  `how_to_apply` say "write a new kernel" (naming the convention/API it
  must match) and record what you searched — the gain is pursued, not
  dropped.
- **Optimizer** — re-run that search before implementing (the roadmap may
  predate your knowledge of the tree). When the item names a kernel to
  wire up but nothing suitable actually exists, fall back to writing the
  kernel rather than recording a no-change blocker — the wiring
  instruction is the item's preferred realization, not its outer scope.
  Whenever you write a new kernel, your summary's *Mapping to the
  roadmap item* section must say what you searched and why nothing fit.
- **Evaluator** — judge a hand-written kernel by whether a suitable
  existing kernel really does exist. When one does, the diff never
  passes the code-quality axis, whatever gain it measures: PUSH_BACK
  with `reason_category: code_quality`, directing the optimizer to wire
  up the existing kernel (REJECT only when out of retries). When the
  attempt's recorded search is right that none exists, the new kernel is
  a legitimate realization — hold it to the normal axes (scoped diff,
  correctness and targeted tests, measured gain), not to a reuse rule it
  cannot satisfy.
"""


# --------------------------------------------------------------------------- #
# Dormant-capability sweep (analyzer)
# --------------------------------------------------------------------------- #

DORMANT_CAPABILITY_SWEEP = """\
## Dormant-capability sweep (round 1)

Profiling only ranks work the build already executes — it is
structurally blind to shipped-but-dormant acceleration surfaces, which
never appear in any trace precisely because they never run. A
multi-token-prediction head sitting unused in the checkpoint, or a
fusion path gated off by an env-var default, can be worth more than
every trace-visible item combined, and no amount of re-profiling will
surface it. Once per campaign, in round 1 **before** authoring the
roadmap, sweep for them explicitly:

1. **Checkpoint config** (`config.json` under `checkpoint_path`):
   speculative-decode / multi-token-prediction heads
   (`mtp_num_hidden_layers`, `num_nextn_predict_layers`, eagle/draft
   blocks) and cache/precision hints — then confirm against the
   checkpoint's weight index that the matching tensors actually ship
   (e.g. `mtp.*` entries in `model.safetensors.index.json`).
2. **Serving config**: knobs the live tuning YAML leaves unset whose
   default disables a capability the checkpoint ships (e.g. no
   `speculative_config` while an MTP layer sits in the weights), and
   documented backend/strategy selectors still on their generic default.
3. **Model code**: env-gated or condition-gated paths in the model's
   modeling file(s) in the checkout that default OFF for this deployment
   shape — `grep -n "environ" <modeling files>` via `Bash`, then read
   each gate's condition against the live config (TP/EP/attention-DP,
   quant mode) to see whether the path could legally run here.

For each dormant surface found, produce exactly one of:

- a **roadmap item** — when its approach is allowed and the mechanism
  plausibly helps this workload. No trace evidence can exist yet, so
  ground `expected_gain_pct` in the SOL projection, the casebook, or the
  mechanism's published behavior instead, and say in `evidence` that the
  lever is dormant plus how you verified it (the config key, the weight
  names, the gate you read);
- or a one-line **dismissal with evidence** — wrong hardware, a gate
  that is provably correct to keep off, a capability incompatible with
  the workload or the campaign's accuracy scope. Never dismiss for "no
  trace evidence" — dormant levers cannot have any.

Record the sweep's outcome under a `## Dormant capabilities` heading in
round 1's `profile_findings.md`: one line per surface found with its
disposition (item id or the dismissal), and "none found" when the sweep
comes up empty. In later rounds re-visit only when an accepted item
changed what is reachable (e.g. a config item just enabled the surface).
"""


# --------------------------------------------------------------------------- #
# Acceptance gate (evaluator)
# --------------------------------------------------------------------------- #

EXPECTATION_GATE = """\
## The acceptance gate — APPROVE, PUSH_BACK, or REJECT

APPROVE the attempt only when **all three** axes below hold. When any
axis fails, choose between the two negative verdicts — either way the
orchestrator reverts every change, so the only question is whether a
retry is worth its benchmark:

- **PUSH_BACK** — the item still looks winnable and you can name a
  concrete, actionable fix: a different knob value, a scoped-diff
  cleanup, a crash with a clear cause, a closer casebook recipe. The
  optimizer retries with your feedback (bounded by
  `optimize.max_attempts_per_item`). On the item's **final attempt**
  PUSH_BACK is unavailable — the orchestrator treats it as REJECT — so
  decide APPROVE or REJECT there.
- **REJECT** — the item's premise is broken and no retry would help:
  the knob does not exist in this build and no faithful variant does,
  the claimed mechanism does not apply to this workload, the optimizer
  reported an unresolvable blocker, or the change regresses for a
  fundamental reason. The item is marked `failed` and the loop moves to
  the next item — a REJECT saves the benchmarks a doomed retry would
  burn, so use it whenever the premise (not the execution) is at fault.

Both negative verdicts carry exactly one `reason_category`. The three
axes:

1. **Code quality** (`reason_category: code_quality` when violated) — the
   diff is minimal and scoped to the roadmap item, follows the
   surrounding code's style, introduces no obvious bugs / dead code /
   debug leftovers, contains no comments or names that reference this
   run's internals (roadmap item ids like `opt-008`, attempts, the perf
   workspace — see *Git discipline*), adds no hand-written kernel whose
   functionality an existing provider already supplies (see *Prefer
   existing kernels over writing new ones*), and for `approach: config`
   changes only documented keys in the tuning YAML.
2. **Functionality** (`reason_category: functionality`) — the server
   starts and serves coherent completions with the change applied. For
   `approach: code` items, also run the narrowest relevant tests in the
   TRT-LLM checkout when a targeted test exists; a server crash, garbage
   output, or a failed targeted test always fails this axis.
3. **Perf expectation** (`reason_category: perf_shortfall`) — measure the
   target metric with the canonical benchmark and compute the gain
   **against `current_best` in `roadmap.yaml`** (the last ACCEPTED
   measurement — never the original baseline; gains accumulate), with
   `accept_fraction` / `noise_floor_pct` from the `optimize` block in
   `task.yaml` and `expected_gain_pct` from the roadmap item.

   **Single operating point** (`benchmark.concurrency` is an integer) —
   compute the gain against `current_best.value`; the attempt passes iff:

   ```
   measured_gain_pct >= accept_fraction × expected_gain_pct
   AND measured_gain_pct >= noise_floor_pct
   ```

   **Curve mode** (`benchmark.concurrency` is a list) — measure **every**
   point, compute one signed per-point gain on the target metric against
   the `current_best.curve` entry with the **same concurrency**
   (direction-normalized per the measurement protocol), then apply the
   **Pareto gate**:

   ```
   gain_i          = per-point gain vs current_best.curve[concurrency = c_i]
   mean_gain_pct   = arithmetic mean of gain_i over the SCORED points
   regression_bar  = optimize.max_regression_pct when task.yaml sets it,
                     else noise_floor_pct

   PASS iff  mean_gain_pct >= accept_fraction × expected_gain_pct
         AND mean_gain_pct >= noise_floor_pct
         AND every gain_i >= -regression_bar   # no point (scored or not) regresses beyond the bar
   ```

   **Regression budget** (`optimize.max_regression_pct`, curve mode only,
   optional): when set, it is an **owner-declared per-point regression
   budget** — the task's explicit decision that a large mean win may cost
   up to that % at individual points (e.g. a saturated top point paying
   for big gains everywhere else). It is never yours to assume: absent
   the key, the bar stays at the noise floor (no real regression is
   acceptable). When an accepted attempt uses the budget — any point
   regresses beyond the noise floor but within the bar — name that point,
   its regression, and the budget in `evaluation.md`'s Verdict and in
   your progress `summary`, so the report can surface the trade
   explicitly rather than bury it in a mean.

   The **scored points** are `optimize.focus_concurrencies` from
   `task.yaml` when set (the regime the campaign optimizes for — an
   unweighted mean over log-spaced points otherwise lets many
   low-throughput points dilute a win in the regime that matters), else
   **all** configured points. Either way you still **measure every
   point** and the no-regress condition covers **every** point — focus
   scoring narrows what the mean rewards, never what a regression can
   veto. When a focus is set, show both means (focus and all-points) in
   `evaluation.md` and state that the focus mean gated.

   Show the per-point rows, the mean(s), and all three conditions in
   `evaluation.md`. Report `measured_gain_pct` = `mean_gain_pct` (the
   **scored** mean), `measured_value` = the mean of the per-point
   absolute values **over the scored points**, and the `curve` field =
   the per-point `{concurrency, value, tok_s_user, tok_s_gpu}` rows for
   **all** points. If `current_best` carries no `curve` (a degraded
   earlier accept), compare your scored mean against `current_best.value`
   for the two mean thresholds, skip the per-point no-regress check, and
   say so in your report.

   In both modes, show the arithmetic in your evaluation report — the
   thresholds, the measured value(s), and the reference value(s).

On APPROVE, `reason_category` is `"none"`. Report `measured_gain_pct` and
`measured_value` in your progress entry **exactly as measured** (signed;
a regression is negative) — the orchestrator records them into
`roadmap.yaml` and advances `current_best` from them, so fabricated or
rounded-up numbers poison every later round.
"""


# --------------------------------------------------------------------------- #
# Measurement protocol (benchmarker / evaluator / qa)
# --------------------------------------------------------------------------- #

MEASUREMENT_PROTOCOL = """\
## Measurement protocol

Comparable numbers are the loop's foundation — every measurement follows
the same recipe:

- Drive the **canonical `benchmark_serving.py` command** at the operating
  point(s) configured in `task.yaml`'s `benchmark` block: ISL / OSL
  fixed, `num_prompts` exactly as configured (a single integer used at
  every point, or a list paired index-by-index with the concurrency
  list — use the paired entry per point), and **one run per
  `benchmark.concurrency` point, sequentially ascending, over one server
  launch** when it is a list (curve mode). Never resize the operating
  point or the point list mid-run — a measurement at a different point
  (or a different set of points, or a different `num_prompts` at the
  same point) is not comparable and is worthless to the loop.
- Pass `--result-dir <the artifact directory named in your instructions>`
  so the result JSON lands next to the stage's other artifacts — in curve
  mode `--result-dir <that directory>/concurrency_<c>` for the run at
  point `<c>` — and read the metrics from that JSON (not from eyeballing
  stdout).
- Metric keys in the result JSON: `output_throughput` (output tok/s — the
  default target metric), `total_token_throughput`, `request_throughput`,
  and latency keys `mean_ttft_ms` / `median_ttft_ms` / `p99_ttft_ms`
  (likewise `*_tpot_ms`, `*_itl_ms`, `*_e2el_ms`). The active target is
  `optimize.target_metric` in `task.yaml`.
- **Direction rule:** throughput metrics are better when higher; `*_ms`
  latency metrics are better when lower. Always report `gain_pct`
  normalized so **positive = improvement**:
  - throughput: `gain_pct = (new − reference) / reference × 100`
  - latency (`*_ms`): `gain_pct = (reference − new) / reference × 100`
- State which reference you compared against (baseline vs current best)
  next to every gain you report. In curve mode, gains are per point
  (same-concurrency reference entry) and aggregate as the **mean** —
  over `optimize.focus_concurrencies` when `task.yaml` sets it (the
  scored subset; still measure and report every point), else over all
  points.

Curve worked example (target `output_throughput`, `expected_gain_pct`
5.0, `accept_fraction` 0.5, `noise_floor_pct` 1.0) —
`current_best.curve`: c=8 → 812.0, c=32 → 1657.0, c=128 → 2210.0;
measured: 846.1, 1755.2, 2201.2. Per-point gains: +4.20%, +5.93%,
−0.40%; mean = +3.24%. Gate: 3.24 ≥ 0.5×5.0 = 2.5 ✓; 3.24 ≥ 1.0 ✓;
worst point −0.40% ≥ −1.0% ✓ → the perf axis passes.
"""


# --------------------------------------------------------------------------- #
# The live tuning config (all server-launching roles)
# --------------------------------------------------------------------------- #

TUNING_CONFIG_NOTE = """\
## The live tuning config (supersedes the `extra_llm_api_options` guidance above)

In this workflow the server tuning is **owned by the workspace**, not by
`task.yaml`: `trtllm-serve` **always** passes
`--extra_llm_api_options <workspace>/tuning/extra_llm_api_options.yaml` —
the live working copy ( `{}` when no tuning applies, which is valid).
Ignore the earlier instruction to pass the flag only when `task.yaml`
sets the key: here the flag is always present and always points at the
tuning file, so config optimizations take effect by editing that one
file and relaunching.

- Only the **optimizer** edits `tuning/extra_llm_api_options.yaml`; every
  other role treats it as read-only and serves with it as-is.
- `tuning/extra_llm_api_options.accepted.yaml` is the
  orchestrator-managed snapshot of the last accepted config — never edit
  it; the orchestrator restores the live file from it when an attempt is
  rejected.
"""


# --------------------------------------------------------------------------- #
# Disaggregated serving (every role that launches or measures a server).
#
# Composed LAST in each role prompt so its overrides win: it supersedes
# the single-server lifecycle, the tuning-config note, and the profiling
# runs above it, the same way TUNING_CONFIG_NOTE supersedes the
# extra_llm_api_options guidance. It is conditional on `task.yaml` — an
# aggregate campaign reads it, finds no `disagg:` block, and ignores it.
# --------------------------------------------------------------------------- #

DISAGG_CAMPAIGN = """\
## Disaggregated serving (supersedes the server-lifecycle, tuning, and profiling guidance above)

**This campaign is disaggregated.** The orchestrator composes this section
only for such a campaign, so it applies unconditionally. You do not launch
`trtllm-serve`, poll `:8000`, or tear a server down — a Slurm job does all
of that. The benchmark command reference above still applies; the harness
runs it for you, with the same flags.

### Inputs

- **harness config** (`task.yaml`'s `disagg.config`) — cluster,
  environment, measurement conditions. Read-only.
- **`task.yaml`** — the campaign knobs (`optimize`). Read-only.
- **`<workspace>/tuning/extra_llm_api_options.yaml`** — the harness
  config's `worker_config`, i.e. `{ctx: {...}, gen: {...}}`. Only the
  **optimizer** edits it, under the rules above.

### Per launch

1. Synthesize: harness config with its `worker_config` replaced by the
   live tuning file, written to `<your artifact dir>/disagg_config.yaml`.
2. Submit and poll in the foreground until the job leaves the queue:
   ```bash
   cd <trtllm_repo_path>/examples/disaggregated/slurm/benchmark
   python submit.py -c <your artifact dir>/disagg_config.yaml \\
       --log-dir <your artifact dir>/bench      # --dry-run to check the node math first
   squeue -j <id>                               # blocking loop; never yield your turn
   ```
3. Read every metric from `<log-dir>/concurrency_<c>/result.json`. On
   failure read `<log-dir>/slurm-<id>.{out,err}` and the per-role worker
   logs before resubmitting — a retry costs another allocation.

The job tears its own cluster down; never kill workers by PID. To abandon
a run, `scancel <id>`.

### Traps that cost an allocation

- `--log-dir` must be a **fresh** path: `submit.py` wipes it if it exists
  without a `trtllm_config.yaml`.
- `environment.work_dir` is where `slurm.script_file` is resolved, **not**
  an output dir. Leave it as the campaign set it.

### `num_gpus` and the frozen topology

`num_gpus` = **sum over roles** =
`num_ctx_servers x (ctx tp x pp x cp) + num_gen_servers x (gen tp x pp x cp)`.

Worker counts and per-role parallel sizes are **frozen for this
campaign**: an attempt that changes one is a REJECT whatever it measured,
and `num_gpus` differing from the baseline's means the comparison is void
— stop and report it. Everything else in the role configs (batch sizes,
token limits, KV-cache, MoE, `cache_transceiver_config`, scheduling,
speculative decoding) is normal `approach: config` work.

### Profiling

nsys only, and the harness owns it — in the config you synthesize:

```yaml
profiling:
  nsys_on: true
  gen_profile_range: <iteration window>   # generation workers
  ctx_profile_range: <iteration window>   # context workers
```

- It wraps **workers only** (never the router or the benchmark client),
  writing `<log-dir>/nsys_worker_proc_<ROLE>_<instance>_<procid>.nsys-rep`.
- Profile generation workers by default — decode is the steady state the
  target metric comes from. Say which role each trace came from.
- Choose each window from the operating point — the roles count
  iterations on different clocks and `profile.nsys_iter_range` is only a
  default. State the windows you used.
- torch profiler and ncu have **no path through this harness**: record
  `not available in a disagg campaign` and plan from nsys — never
  fabricate a trace.
- KV-cache transfer (ctx to gen) is a first-class cost here that an
  aggregate campaign does not have; classify it as `communication`.
"""


# --------------------------------------------------------------------------- #
# Actionable casebook variant (optimizer)
# --------------------------------------------------------------------------- #

CASEBOOK_APPLY = """\
## Apply from the optimization casebook (load it early)

Before implementing the roadmap item, **load the
`perf-optimization-casebook` skill** with the `Skill` tool and keep it
open as reference. It ships with the `trtllm-agent-toolkit` plugin, so
invoke it as `perf-optimization-casebook` — or the fully-qualified
`trtllm-agent-toolkit:perf-optimization-casebook` if the bare name is not
found. Do this **early in your turn, right after you read the roadmap
item**, so the implementation is anchored to a proven precedent instead
of improvised.

When the roadmap item names a `casebook_ref` (or its signal matches a
case), implement from that case — its how-to-apply steps, its
accuracy-risk notes, its verification and rollback guidance — adapting
the proven recipe to this config rather than inventing a new one. Where
the item and the casebook disagree, the roadmap item's `how_to_apply`
wins (it is grounded in this run's profiling evidence); note the
divergence in your summary.

If the skill is not available in this environment, note that in one line
and implement from the roadmap item's `how_to_apply` alone — never block
the run on it.
"""


# --------------------------------------------------------------------------- #
# HTML companion (reporter) — adapted from perf-analyze's HTML_COMPANION
# --------------------------------------------------------------------------- #

OPTIMIZE_HTML_COMPANION = """\
## HTML companion (`optimization_report.html`)

Produce a **single self-contained** HTML file alongside the markdown — all
CSS/JS inline, **no external CDN, font, or asset URLs** so it opens
offline. It presents the *same content* as `optimization_report.md` (same
sections, same numbers, same outcomes) in a clean, interactive form.

**Required structure (top-down):**

1. `<!DOCTYPE html>` with `<html lang="en">`, a `<title>` matching the
   report's H1, and `<meta name="viewport">`.
2. Inline `<style>`: clean readable font stack, generous line-height,
   ~800–900 px max content width, and light/dark mode via
   `@media (prefers-color-scheme: dark)`.
3. A **sticky table-of-contents nav** listing every H2, each linking to
   the section's slugified anchor id (`#executive-summary`, etc.).
4. The main `<article>` body, sections in the same order as the markdown
   (Executive Summary, Configuration, Baseline, Optimization Trajectory,
   Pareto Improvement — curve mode only, Applied Optimizations,
   Kernel-Level Comparison, Failed Attempts, Final Verification, Config
   & Code Diff Summary, Remaining Roadmap, Durable facts for the next
   campaign), each heading carrying a stable id.
5. Metric tables are real HTML `<table>`s (same columns/values as the
   markdown). The **Executive Summary**'s cumulative-improvement headline
   is visually prominent (e.g. a callout box).
6. Inline `<script>` at the end of `<body>`.

**Required charts (self-contained — no chart library, no CDN):** embed
each chart's data as a JSON array in the inline script and render it to
inline SVG with your own small renderer. Style via CSS variables so both
color schemes stay readable, and never plot a value that differs from
the section's table — the table is the source of truth.

- **Trajectory line chart** — at the top of *Optimization Trajectory*:
  x = the steps in applied order (baseline → each accepted item → the
  final verification, when it ran), y = the target metric (curve mode:
  the **mean across concurrency points** — label the y-axis
  accordingly). Mark every point and label
  it with its item id, draw a dashed horizontal reference at the
  baseline value, pad the y-domain around the data (do not force zero),
  and put the metric name + units on the y-axis. Each point gets a hover
  tooltip (an SVG `<title>` is enough) showing the absolute value and
  the cumulative gain. With fewer than two points, keep the table and
  omit the chart.
- **Pareto improvement chart** — at the top of *Pareto Improvement*,
  only in curve mode: **x = tok/s/user, y = tok/s/gpu**, exactly **two
  polylines — the baseline curve vs the final curve** (final = the
  final verification's qa entry `curve`; fall back to
  `current_best.curve` and say so in the section). Mark every point, label it `c=<n>`, give it an SVG
  `<title>` tooltip with the exact x/y values, distinguish the two
  series by more than hue alone (e.g. solid vs dashed) plus a legend,
  and pad both axis domains around the data. Intermediate accepted
  items are **not** plotted (their curves stay in progress.yaml for
  audit). When the report also carries per-point **SOL-projected**
  values (the projector ran in curve mode and `sol_projection.md`
  holds a per-point ceiling table), overlay the projected ceiling as a
  third polyline, distinguished from both measured series by more than
  hue alone (e.g. dotted) plus its own legend entry — omit the overlay,
  never approximate it, when the projection is missing or unavailable.
  Omit the chart and the section in scalar mode or when either
  curve is missing.
- **Kernel before/after bars** — at the top of *Kernel-Level
  Comparison*, only when both profiles exist: paired horizontal bars
  (before vs after GPU-time share) per kernel in the comparison table,
  the two series distinguished by more than hue alone (e.g. solid vs
  outlined) plus a legend naming each profile's round.

**Required interactivity:**

- **TOC scroll-spy** — the entry for the section in view gets an `active`
  class as the reader scrolls.
- **Collapsible H2 sections** — clicking a heading toggles a `collapsed`
  class on its body; default expanded.
- **Print-friendly** — hide the TOC and force-expand all sections in
  `@media print`.

**Faithfulness rule:** the HTML is not a remix — same sections, same
tables, same expected-vs-measured numbers and outcomes as the markdown,
and charts that plot exactly the numbers in the tables they sit above.
If you revise the markdown, revise the HTML in the same turn.
"""


# --------------------------------------------------------------------------- #
# SOL projection consumption (appended only when the projector stage is enabled)
# --------------------------------------------------------------------------- #

SOL_ANALYZER_CONTEXT = (
    """\
## SOL projection as context (the projector stage ran)

The Projector ran once, after the baseline benchmark, and left
`sol_projection.md` in the workspace — an analytical speed-of-light
(SOL) ceiling for this model/hardware/operating point, derived with the
`internal-perf-sol-analysis` skill, with a baseline-vs-SOL gap analysis.
`Read` it (or call `read_latest_progress` with `agent: "projector"`)
after `baseline/benchmark_results.md` and use it as **context, not
evidence**:

- Let the projected headroom (% of SOL) and the bound mix (compute /
  memory / launch) inform which casebook families you prioritize and
  how you rank roadmap items — e.g. a low % of SOL with a memory bound
  raises the prior on memory-bandwidth items; a gap far beyond what the
  ceiling can explain points at host/scheduling overhead (the ceiling
  models kernel execution plus per-launch latency only, so
  serving-stack scheduler and queueing costs are invisible to it).
- **Sanity-bound `expected_gain_pct` against the ceiling**: an item
  cannot plausibly recover more than the SOL headroom says is available
  on its bound side — an estimate that would push the metric past the
  ceiling is over-promised; tighten it and say so in
  `expected_gain_rationale`.
- The ceiling stays valid across rounds (it is a property of the
  hardware + model + operating point, not of the optimizations you
  apply). Its measured-vs-SOL table is the **baseline** snapshot: in
  round N > 1 compare your fresh measurements against the same ceiling
  — never re-derive it, never treat its measured column as current.
- Measured trace evidence always outranks the projection: when they
  disagree, trust the trace and note the disagreement. In
  `profile_findings.md`, say where the profile **confirms or
  contradicts** the projection (a sentence per ranked hypothesis is
  enough).
- **No silent exhaustion — account for the gap you leave behind.** In
  any round where you leave `roadmap.yaml` with no actionable pending
  item while the projection says meaningful headroom remains, close
  `profile_findings.md` with a **## Remaining-gap attribution** section:
  decompose the remaining gap-to-SOL into named parts (from the
  projection's bound mix, your fresh profile — the correlation table's
  per-op gap rows are the natural part names — and the failed items'
  `evaluation.md` evidence — their *Gap implication* lines) and give
  every part exactly one of: a **new roadmap item** that attacks it, or
  an **evidence-backed reason it cannot be closed in this campaign**
  (mechanism already present in the build, needs a rebuild the campaign
  cannot do, accuracy risk the task forbids, allowed-approach
  restriction, host/scheduler cost outside the ceiling's model — cite
  the artifact, not a hunch). A part no evidence explains is recorded
  as **unexplained**, never absorbed into the other buckets. The
  Reporter's remaining-gap accountability is built from this section —
  the campaign must never end with headroom that is neither attacked
  nor accounted for.
- Projected numbers are not measurements — never present a SOL number
  as a measured one. If `sol_projection.md` is missing or declares
  itself unavailable, ignore it for ranking, skip the correlation
  below, and record that in *Caveats*.

"""
    + SOL_CORRELATION_METHOD
    + """
Artifact placement in this workflow: `regions.json`, `sol.json`, and
any `sol_recipes/` go in **this round's `analysis/` directory**; the
peaks file stays campaign-level at `<workspace>/sol_work/peaks.json`.
Re-run the correlation **every round that profiles** against that same
peaks file — the ceilings do not move, your measured rows do, so the
fresh per-op table is what re-ranks pending items, sanity-bounds their
`expected_gain_pct` per op, and names the parts of any *Remaining-gap
attribution*. A replan-only round produced no new measured rows: read
the standing correlation, do not re-derive it.
"""
)


SOL_OPTIMIZER_CONTEXT = """\
## SOL projection as context (the projector stage ran)

The Projector ran once, after the baseline benchmark, and left
`sol_projection.md` in the workspace — an analytical speed-of-light
(SOL) ceiling for this model/hardware/operating point, with a
baseline-vs-SOL gap analysis naming which ceiling binds per phase
(compute / memory / launch, plus comm on multi-GPU). `Read` it
alongside the roadmap item and use it as **context, not spec**:

- **Aim the implementation at the binding ceiling.** Where the item's
  `how_to_apply` leaves you a choice — between realization variants,
  knob values, or kernels — prefer the one that attacks the bound the
  projection names for the phase the item targets: memory-bound → move
  fewer bytes (fusions that eliminate round trips, lower-precision I/O,
  better layouts); launch-bound → fewer or amortized launches;
  compute-bound → higher-throughput math paths.
- Add one line to `optimization_summary.md`'s *Mapping to the roadmap
  item* section — `SOL alignment: <the bound this change attacks and
  the mechanism (bytes moved / launches / math throughput)>` — so the
  Evaluator's kernel evidence and the final report can check the
  claimed mechanism against the trace.
- **The projection never expands the item.** Implement exactly the
  roadmap item — unclaimed headroom is the Analyzer's to plan into new
  items, not yours to chase in this attempt.
- The item's `how_to_apply` and the measured evidence behind it outrank
  the projection when they disagree. If `sol_projection.md` is missing
  or declares itself unavailable, proceed without it and skip the
  `SOL alignment` line.
"""


SOL_OPTIMIZE_REPORTER_GUIDANCE = """\
## Projection vs Measured (the projector stage ran)

The Projector ran once, after the baseline benchmark, and left
`sol_projection.md` — an analytical speed-of-light (SOL) ceiling
derived with the `internal-perf-sol-analysis` skill, with % of SOL /
MFU / MBU numbers and a baseline-vs-SOL gap analysis. `Read` it with
the other inputs and add one section to `optimization_report.md`,
placed **between "Final Verification" and "Config & Code Diff
Summary"** (the HTML companion mirrors it like every other section):

```
## Projection vs Measured

<A headroom-captured table for the target metric (plus throughput /
TTFT / TPOT rows where the projection carries them):
| Metric | Baseline | Final | SOL | Baseline % of SOL | Final % of SOL |
In Pareto-curve mode the table gains a leading concurrency column and
one row-group per configured point, pairing with sol_projection.md's
per-point tables. Follow with the headline: how much of the projected
headroom the campaign captured (final % of SOL − baseline % of SOL) and
the projected bound mix (compute / memory / launch).

Close the section with **Remaining-gap accountability** — why the
campaign ends with the gap it has. Decompose the remaining gap-to-SOL
into named parts and give every part exactly one verdict, each backed
by a cited artifact (a failed item's `evaluation.md` and its *Gap
implication* line, a round's `profile_findings.md` remaining-gap
attribution, the projection's own caveats):

| Gap part | ~share of gap | Verdict | Evidence |

- `closed` — headroom the campaign captured (cite the accepted items);
- `infeasible: <constraint>` — provably not closable in this campaign:
  mechanism already present in the build, needs a rebuild the campaign
  cannot do, accuracy risk the task forbids, allowed-approach
  restriction, host/scheduler cost outside the ceiling's model — name
  the evidence, never a hunch;
- `untried` — plausibly closable but never attempted (round/attempt
  budget ran out first) — the follow-up campaign's opening move;
- `unexplained` — headroom no artifact accounts for; report it as a
  finding, never absorb it into the other buckets.>
```

A campaign that accepted nothing must still fill the accountability
breakdown — one sentence on why **zero** projected headroom was
captured (which buckets the whole gap fell into), echoed in the
Executive Summary.

Weighing rules:
- **SOL numbers come only from `sol_projection.md`** — never re-derive
  or extrapolate them. The baseline side comes from the projection's
  own measured-vs-SOL table; the final side comes from the final
  verification's independent measurement, else the roadmap ledger
  (`current_best`) — say which. When the final verification did not run
  (nothing accepted), final == baseline and the section must say the
  campaign captured none of the projected headroom.
- **Echo the headroom story where it matters**: one sentence in the
  Executive Summary (how much of the SOL headroom was captured, and —
  when most of it was not — the dominant accountability bucket why) and
  one in Remaining Roadmap (which bound class the remaining gap-to-SOL
  sits in, per the projection — corroborated or corrected by the last
  round's profile).
- **Every accountability verdict traces to an artifact.** The gap parts
  and their verdicts come from the failed items' `evaluation.md` (their
  *Gap implication* lines), the rounds' `profile_findings.md`
  (the *Remaining-gap attribution* section when the analyzer wrote
  one, and the *SOL correlation* per-op table — its largest-gap
  regions name the parts), and the projection's own caveats. A part
  you cannot back with a citation is `unexplained` — writing a
  plausible-sounding justification for it is worse than reporting it
  unexplained.
- The projection is a model, not a measurement — when it conflicts with
  measured evidence, measured evidence wins, and the conflict is worth
  a sentence.
- The ceiling models kernel execution plus per-launch latency only — a
  measured result far below it often indicates serving-stack
  scheduler/queueing costs the model does not price; treat that as
  supporting evidence for host-side categories, not as a contradiction.
- If `sol_projection.md` is missing or declares itself unavailable, the
  section must honestly say **"Projection unavailable (<reason>)"** and
  the report falls back to measured evidence alone — never fabricate
  projected numbers.
"""


# --------------------------------------------------------------------------- #
# Per-kernel coverage contract (analyzer / reporter) — built per run, only
# when task.yaml declares profile.kernel_coverage
# --------------------------------------------------------------------------- #


def kernel_coverage_analyzer_note(min_share_pct: float, coverage_target_pct: float) -> str:
    """The analyzer's per-kernel coverage contract, with the task's bars.

    Appended only when ``task.yaml`` declares ``profile.kernel_coverage``.
    It supersedes Run C's top-kernel target selection with coverage-driven
    enumeration, poses the two per-kernel questions (faster? fusible?),
    and defines ``kernel_ledger.yaml`` — the machine-readable proof,
    validated by the orchestrator each round, that every enumerated
    kernel's optimization and fusion possibility was considered.
    """
    return f"""\
## Per-kernel coverage contract (this task declares `profile.kernel_coverage`)

This campaign carries an exhaustiveness guarantee: **every kernel
at/above the coverage bar gets an ncu SOL deep dive and an explicit
answer to two questions — (1) can this kernel be made faster? (2) can it
be fused with its neighbors? — each answer a roadmap item or an
evidence-backed dismissal.** You record the answers in
`kernel_ledger.yaml` (contract below) in this round's `analysis/`
directory, every round **that profiles**. The orchestrator
schema-validates the ledger the moment your turn ends — a missing row,
an unanswered question, an `item` ref that matches no roadmap id, or
coverage below the target **aborts the stage**, exactly like an invalid
roadmap. The campaign cannot conclude with a hot kernel nobody looked
at.

A round your instructions open as **replan-only** (or as a reused
analysis) ran no ncu and is exempt: it owes no ledger, and the
orchestrator waives the contract for it rather than aborting over an
artifact the round was told not to produce. The standing ledger still
describes that build — the round changed nothing about it.

### Coverage-driven ncu targeting (supersedes Run C's target selection)

Run C's "top 3–6 stems, never profile every kernel blindly" rule is
superseded — this task pays for breadth:

- **Enumerate from the fresh nsys `cuda_gpu_kern_sum`**: every kernel at
  or above **{min_share_pct}%** of profiled GPU time gets a ledger row;
  when those rows sum below **{coverage_target_pct}%**, keep taking the
  next-largest kernels until the target is covered. Roll everything
  below the cut into a single explicit `other` share — recorded, never
  silently dropped.
- **Group where the disposition is genuinely shared**: closely related
  kernels (e.g. a family of small elementwise/cast variants between the
  same producers and consumers) may share one row, with the members
  named in `full_name` and their summed share in `share_pct`. Grouping
  is for honest shared verdicts — never to bury a kernel whose answer
  would differ.
- **Capture ncu in bounded passes, not one blind sweep.** One pass's
  `--launch-count` is consumed in launch order, so per-layer hot kernels
  exhaust it before once-per-step kernels (final norm, logits GEMM,
  sampler) ever match. Run Run C's canonical command up to **3 passes**:
  pass 1 filters on the hottest stems exactly as Run C describes; then
  check which enumerated stems the report actually captured (`ncu
  --import ... --page raw --csv`), and each further pass filters on
  **only the still-missing stems** (so its budget is spent on them),
  with `--launch-count` ≈ 8 × that pass's stem count (cap ~300). Name
  the artifacts `server_ncu_pass<k>.ncu-rep` (+ per-pass
  `ncu_details_pass<k>.txt` / `ncu_raw_pass<k>.csv`); each pass is its
  own server relaunch, gated to the same iteration window.
- **Degrade honestly, never fabricate**: a kernel no pass captured (or
  ncu itself unavailable) keeps its ledger row with
  `ncu: "unavailable: <reason>"` — both questions are still owed,
  answered from the nsys timeline, the torch trace, and the source.

### Question 1 per kernel — can it be made faster?

Classify the kernel with the `perf-nsight-compute-analysis` skill's
thresholds and take the levers for that class from its bottleneck
guide. What the guide cannot tell you is where they live in this
codebase:

- **memory-bound** → the round trip may be removable outright, which is
  question 2.
- **compute-bound** → a better kernel/backend for the shape usually
  already exists (the checkout's backend selectors, flashinfer,
  provider GEMMs); lower-precision math only where the task's accuracy
  scope allows.
- **latency-bound** → most often a CUDA-graph / launch-amortization
  roadmap item rather than a kernel edit.
- Whatever the class, run the *Prefer existing kernels* search first —
  the faster variant usually already ships somewhere in the checkout or
  its providers.

A `dismissed` answer must name its evidence. The recurring legitimate
dismissals — lead the `ref` with the matching tag when one fits:

- `at-sol-floor: <side> SOL <n>%` — the binding side already runs at
  ~≥85% of its ceiling; nothing material left in this kernel alone.
- `below-materiality: <share>% × best-case recovery < noise floor` —
  show the arithmetic against `optimize.noise_floor_pct`.
- `needs-rebuild: <artifact>` — the lever lives in compiled artifacts
  this campaign cannot rebuild (cite how you verified). Being unable to
  rebuild the incumbent does **not** by itself close the kernel: the
  campaign can still write a *replacement* kernel (Triton / CuTe DSL /
  an inline-compiled extension — the *Prefer existing kernels* fallback)
  and reroute the Python call site to it. The tag is legitimate only
  when the replacement path is also ruled out: no Python-reachable
  dispatch point to reroute (the launch is internal to a compiled op
  you cannot intercept), or the incumbent is a tuned provider kernel
  near enough its bound-class ceiling that a hand-written kernel has no
  credible headroom (say which, with evidence). Otherwise the answer is
  an item, not this tag: write the new kernel and swap the call site —
  the accept gate already ensures the swap only lands if it measures
  faster.
- `approach-restricted` / `accuracy-scope` — the only lever needs a
  disallowed approach or a lossy change the task forbids; record the
  insight under *Out-of-scope opportunities* in the findings and point
  the ref there.

### Question 2 per kernel — can it be fused with its neighbors?

Fusion verdicts rest on **observed adjacency, not guesses**. Derive each
kernel's neighborhood from the traces: the launch sequence inside one
steady-state step (`nsys stats --report cuda_gpu_trace`, or the
timeline around the kernel's instances) gives the predecessor/successor
kernels; the torch trace's op attribution gives the producer/consumer
tensors between them. Record it in the row's `fusion.neighbors`. Then
test the candidate patterns:

- elementwise/cast/activation chains between two anchors → one fused
  kernel or the producer's epilogue;
- norm + quantization, RoPE + KV-cache write, dequant + GEMM
  prologue/epilogue, attention-adjacent glue;
- the *existing kernels first* rule applies — a fusion item's
  `how_to_apply` names the shipped fused op to wire up when one exists,
  and a new kernel is the encouraged realization when none does.

Judge materiality on the **whole chain**, not the single kernel: a
0.6% kernel between two 0.5% neighbors in one fusible chain is a ~1.6%
opportunity. The recurring legitimate dismissals:

- `multi-consumer-pinned` — the intermediate feeds >1 consumer, so
  fusion cannot remove the round trip (cite the consumers from the
  torch trace / source).
- `already-fused` — the kernel is itself the fused form of its
  neighborhood; nothing adjacent left to absorb.
- `phase-boundary` — the neighbors sit across a CUDA-graph capture,
  stream, or prefill/decode phase boundary a fusion cannot cross.
- `neighbors-at-bandwidth-floor` — every byte both sides move is
  mandatory model/KV traffic; fusing saves no traffic (show the bytes).
- `below-materiality` — the whole chain's share × best-case saving is
  under the noise floor (show the arithmetic).
- `needs-rebuild` — same bar as question 1's tag: "absorbing the
  neighbor means editing a compiled kernel" dismisses the fusion only
  when a *newly written* fused kernel replacing the incumbent plus its
  glue is also ruled out (no reroutable call site, or no credible
  headroom over the tuned incumbent — with evidence).

### The kernel ledger contract (`kernel_ledger.yaml`)

Write one ledger per round into this round's `analysis/` directory. Its
exact shape:

```yaml
version: 1
source: rounds/round_<n>/analysis/nsys_stats.txt   # the kern_sum you enumerated
coverage:
  enumerated_share_pct: 96.8    # sum of kernels[].share_pct
  other_share_pct: 3.2          # the explicit below-bar tail (they must total ~100)
  min_share_pct: {min_share_pct}
kernels:                        # descending share_pct; one row per kernel/group
  - kernel: gdn_bf16_state              # distinctive stem or group label (unique)
    full_name: "void tensorrt_llm::..." # representative full name(s); group members
    share_pct: 18.4                     # % of profiled GPU time (nsys kern_sum)
    ncu:                                # metrics mapping (or the string below)
      duration_us: 41.2
      sm_sol_pct: 12.1
      mem_sol_pct: 78.5
      occupancy_pct: null               # a metric the capture did not yield is null
      bound: memory                     # compute | memory | latency | balanced | comm
      note: "occupancy section empty: replay stalled"   # required by that null
    faster:
      disposition: item                 # item | dismissed
      ref: opt-003                      # roadmap item id | evidence-backed dismissal
    fusion:
      disposition: dismissed
      neighbors: "rmsnorm -> THIS -> fp8_quant (cuda_gpu_trace, step 120)"
      ref: "multi-consumer-pinned: intermediate feeds residual add + next norm (torch_trace)"
  - kernel: allreduce_fusion            # a collective: never goes under ncu at all
    full_name: "void tensorrt_llm::kernels::ar_fusion::..."
    share_pct: 9.2
    ncu: "unavailable: collective — kernel replay deadlocks the ranks"
    bound: comm                         # with the string form, `bound` sits here
    faster:
      disposition: dismissed
      ref: "approach-restricted: strategy A/B falsified in a prior round; no NVLS here"
    fusion:
      disposition: dismissed
      neighbors: "sigmoid_gate_mul_add -> THIS -> scaleMatrixPerTensorVec (step 120)"
      ref: "already-fused: this IS the AR + residual/norm/quant fused epilogue"
```

Rules:

- **Both questions, every row.** `disposition: item` refs a roadmap item
  id — one existing already, or one you author this round; several rows
  may share one item (a fusion item covers every kernel it merges), and
  the referenced item may already be `accepted`/`failed` (the
  possibility *was* considered — that is the point). `disposition:
  dismissed` carries the evidence in `ref`, tagged per the vocabularies
  above, citing the artifact (an ncu row, the torch trace, a source
  file, a failed item's `evaluation.md`).
- **Say "not measured", never guess it.** A collective never goes under
  `ncu` — kernel replay deadlocks it — so disposition an allreduce from
  its nsys share and the source, give `ncu` the `unavailable: <reason>`
  string, and record `bound: comm` **on the row, beside `ncu`**. When a
  pass reaches a kernel but a section comes back empty, null that metric
  and say why in `note`, rather than fabricating a percentage or
  throwing away the numbers you did measure. `bound` is the one field
  always owed, and the schema enforces it in both shapes: inside `ncu`
  when `ncu` is a metrics mapping, on the row when `ncu` is the degrade
  string. `neighbors` is the evidence a fusion *dismissal* rests on — a
  fusion `item` carries its adjacency in the roadmap entry `ref` names.
- **An unactionable item is not an answer.** Do not park a kernel on an
  item whose `expected_gain_pct` sits below `optimize.noise_floor_pct`
  (the orchestrator never dispatches it) — that is a
  `below-materiality` dismissal wearing an item costume.
- **Round N > 1**: author a fresh ledger from the fresh profile. Carry a
  dismissal forward only when the kernel is unchanged (share within
  ~20% relative, same bound class, no accepted item touched it) — cite
  the original round's evidence plus `carried from round <k>`;
  re-derive every row an accepted item changed, and give fresh rows to
  kernels that newly crossed the bar.
- **Mirror it for humans**: add a `## Kernel disposition ledger` section
  to `profile_findings.md` — the same rows as a table (kernel, share,
  bound, faster →, fusion →) with a one-line rationale each, marking
  every row whose `bound` did *not* come from an ncu capture (the
  degrade string, or a null metric's `note`) so the table cannot be read
  as more measured than it is. The YAML file is authoritative; the
  findings section carries the prose.
"""


KERNEL_COVERAGE_REPORTER_GUIDANCE = """\
## Kernel Coverage (this task declares `profile.kernel_coverage`)

Every analyzer round wrote a `kernel_ledger.yaml` into its `analysis/`
directory — one row per kernel at/above the task's share bar, each
answering *faster?* and *fusible?* with a roadmap item or an
evidence-backed dismissal. `Read` the **final round's** ledger (your
instructions name it) and add one section to `optimization_report.md`,
placed **between "Kernel-Level Comparison" and "Failed Attempts"** (the
HTML companion mirrors it like every other section):

```
## Kernel Coverage

<Open with the coverage headline: N kernels/groups enumerated covering
X% of profiled GPU time (the explicit `other` tail Y%), from round <n>'s
ledger. Then the accountability table, one row per ledger kernel in
descending share:
| kernel | share % | bound | faster → | fusion → |
where each `→` cell resolves the ledger disposition to its campaign
outcome: an item ref becomes `<item-id>: accepted +X%` / `failed
(<reason_category>)` / `pending at campaign end` (from roadmap.yaml);
a dismissal shows its leading tag (`at-sol-floor`, `below-materiality`,
`multi-consumer-pinned`, ...) — keep the full evidence one click away in
the ledger rather than inflating the table. Close with the coverage
accountability sentence: every enumerated kernel had both questions
answered; itemize the rows whose item was still `pending` when the
budget ran out — the untried tail a follow-up campaign starts from —
and mirror those into Remaining Roadmap / Durable facts (`[alive]`).>
```

Rigor rules for this section:

- **Every cell traces to the ledger or the roadmap** — dispositions and
  share values come from `kernel_ledger.yaml` verbatim; outcomes come
  from `roadmap.yaml` statuses and `measured_gain_pct`. Never re-derive
  or soften a dismissal.
- **The final round's ledger is the coverage proof.** Earlier rounds'
  ledgers are history (cite one only to show how a disposition
  evolved); the guarantee the section attests is the final state's.
- **Say how much of the table ncu actually measured.** A row whose `ncu`
  is the `unavailable: <reason>` string, or whose metrics are null with
  a `note` explaining the gap, was dispositioned from nsys and the SOL
  correlation — not from a capture. Count those rows, state it in the
  headline ("ncu contributed per-kernel metrics for 3 of 22 rows; the
  rest carry the ledger's degrade reason"), and qualify each such
  `bound` cell with the ledger's reason (`memory — no ncu: replay
  stalled`). A coverage proof built on unmeasured rows must never render
  like one built on measured rows.
- If the final round's ledger is missing or invalid, say so plainly
  ("Kernel coverage ledger unavailable (<reason>)") — never reconstruct
  rows from memory.
"""


# --------------------------------------------------------------------------- #
# Approach restriction (analyzer / optimizer / evaluator) — built per run
# --------------------------------------------------------------------------- #

_APPROACH_GUARDS = {
    "config": """\
  - `tuning/extra_llm_api_options.yaml` is **read-only for every role**
    this run. The orchestrator compares it against the accepted snapshot
    after every optimizer attempt and **auto-rejects the attempt without
    any evaluation** when it changed. Realizing a config knob through a
    source edit instead (changing a default value, an env-var fallback)
    is the same violation in disguise — don't.\
""",
    "code": """\
  - The TRT-LLM checkout is **read-only for every role** this run. The
    orchestrator checks `git status --porcelain` after every optimizer
    attempt and **auto-rejects the attempt without any evaluation** when
    the worktree is dirty.\
""",
}


def approach_restriction_note(allowed: Sequence[str]) -> str:
    """The prompt block for a run restricted to a subset of ``APPROACHES``.

    Returns ``""`` when every approach is allowed (nothing to say). The
    block is appended to the analyzer / optimizer / evaluator prompts —
    each role gets its consequence of ``optimize.approaches`` spelled
    out, mirroring the deterministic enforcement in the orchestrator
    (item-selection filter + post-optimizer auto-reject).
    """
    allowed = tuple(a for a in APPROACHES if a in allowed)
    disallowed = tuple(a for a in APPROACHES if a not in allowed)
    if not disallowed or not allowed:
        return ""
    allowed_str = ", ".join(f"`{a}`" for a in allowed)
    disallowed_str = ", ".join(f"`{a}`" for a in disallowed)
    guards = "\n".join(_APPROACH_GUARDS[a] for a in disallowed)
    return f"""\
## Approach restriction (`optimize.approaches`)

`task.yaml` restricts this run to `optimize.approaches:
[{", ".join(allowed)}]`: only {allowed_str} roadmap items may be
planned, applied, or accepted; {disallowed_str} is off-limits. What this
means per role:

- **Analyzer** — every roadmap item's `approach` must be one of the
  allowed values. When profiling exposes an optimization that would need
  a disallowed approach, do **not** add it to `roadmap.yaml`; record it
  under an "Out-of-scope opportunities" heading in `profile_findings.md`
  instead, so the insight is preserved without planning unactionable
  work.
- **Optimizer** — never implement an item through a disallowed approach,
  and never work around the restriction:
{guards}
- **Evaluator** — an attempt whose diff works through a disallowed
  approach never passes the code-quality axis, whatever gain it
  measures: PUSH_BACK with `reason_category: code_quality` toward the
  allowed approach(es), or REJECT when the item cannot be realized
  through an allowed approach at all.
"""
