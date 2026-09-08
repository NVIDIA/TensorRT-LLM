---
name: perf-optimize
description: Launch and operate this repo's perf-optimize workflow, which iteratively APPLIES TensorRT-LLM serving optimizations — baseline benchmark at one concurrency or a Pareto curve of them (tok/s/user vs tok/s/gpu), analytical SOL projection on by default (via the internal-perf-sol-analysis skill) sizing the headroom the campaign chases, profile-ranked roadmap.yaml (nsys + torch-profiler + ncu per-kernel analysis via the perf-nsight-compute-analysis skill), a fixed budget of rounds applying items one at a time gated on measured gain (curve mode uses a Pareto gate; the evaluator accepts, rejects, or pushes back each attempt and nsys-profiles every accept), one final-verification QA benchmark, expected-vs-measured report with Pareto improvement results. Use when the user wants to optimize / improve / speed up a trtllm-serve deployment (throughput, TTFT, TPOT, ITL, e2e latency) or says "run perf-optimize". For diagnosis WITHOUT applying changes, use the perf-analyze workflow instead.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Operating the perf-optimize workflow

`perf-optimize` is this repo's iterative optimization campaign for
`trtllm-serve`: benchmarker (baseline) → projector (a one-shot
analytical speed-of-light ceiling; on unless task.yaml sets
`sol.enabled: false`) → a fixed budget of rounds of
[analyzer (re-profile when the runtime evidence is stale; otherwise
replan-only) → (optimizer ⇄ evaluator) per roadmap item] → one
final-verification QA pass → reporter. Every change is gated on code
quality, functionality, and measured gain vs the last accepted
measurement — the evaluator's verdict is three-way (APPROVE / REJECT
terminally / PUSH_BACK for a bounded retry), and every accept is
profiled under nsys (accept-evidence capture). No agent decides when to
stop: the loop runs `optimize.max_rounds` rounds unless the roadmap
runs out of actionable items or the optional improvement target is met.
The deliverable is `<workspace>/optimization_report.md` (+ `.html`).

Do not hand-roll a serve/benchmark/tune loop when the user asks to
optimize serving performance — drive this workflow instead. Authoritative
references (read them before answering detailed questions):

- `<agent_flow>/workflows/perf_optimize/README.md` — contract, acceptance
  gate, workspace layout, git requirements.
- `<agent_flow>/workflows/perf_optimize/task.example.yaml` — fully commented
  task template.

Use perf-optimize when the user wants changes applied and verified.
The workflow itself now ships from TensorRT-LLM, under `agent-flow/`;
the paths below are relative to the INSTALLED `agent_flow` package. Print
its location with
`python -c "import agent_flow,pathlib;print(pathlib.Path(agent_flow.__file__).parent)"`,
or read them out of `<trtllm_repo_path>/agent-flow/agent_flow/`.

Use `perf-analyze` (same repo) when they only want a bottleneck diagnosis.

## 1. Preflight — verify before launching

Run these checks; fix or ask only where noted.

1. **CLI installed**: `perf-optimize --help` works. If not:
   `pip install -e <trtllm_repo_path>/agent-flow`. The workflow drives
   the Claude Code backend, so the `claude` CLI must be installed and
   signed in; `CLAUDE_CODE_DEFAULT_MODEL` overrides the model if the
   user wants.
2. **Paths from the user**: `checkpoint_path` (model checkpoint dir) and
   `trtllm_repo_path` (TensorRT-LLM checkout) must exist — the CLI
   refuses to start otherwise.
3. **Git safety (STOP if it fails)**: `trtllm_repo_path` must be a
   TensorRT-LLM git repo, and its worktree should be clean
   (`git -C <repo> status --porcelain` empty). The workflow mutates the
   checkout: rejected attempts are reverted with `git reset --hard` +
   `git clean -fd`, which destroys any uncommitted, unignored changes
   present when the run starts. If the worktree is dirty, do NOT launch —
   ask the user to commit/stash (or confirm the changes are disposable).
   Work happens on a fresh `perf-optimize/<workspace>-<timestamp>` branch
   off the current HEAD; one commit per accepted item.
4. **Editable install (affects scope, not launch)**: source-level
   (`approach: code`) items only take effect when
   `python -c "import tensorrt_llm; print(tensorrt_llm.__file__)"`
   resolves into `trtllm_repo_path`. If it doesn't, tell the user the run
   will be config-only (the agents detect this too) — still fine to run.
5. **Environment**: either the current node has GPUs (`nvidia-smi`
   works), or the task carries a `slurm-environment` block that routes
   the server + benchmark through a Slurm-launched container
   (`task.example.yaml:139`). On a login or head node with no local
   GPUs — the most common case — bring up the node and container with
   your own site's recipe first.
6. **Internal toolkit skills (affects depth, not launch)**: the
   default-on SOL projector wants `internal-perf-sol-analysis`. That is
   an `internal-`prefixed skill, so **open-source builds of the
   `trtllm-agent-toolkit` plugin strip it** while keeping
   `perf-analysis`. **The CLI checks this itself** at launch and prints
   one line when the skill is missing, so this is context for the user
   rather than something you must gate on. The campaign completes either
   way: without the SOL skill the projector falls back to
   `perf-analysis`, grounds the peaks from named sources rather than the
   skill's calculator, marks them as such, and writes no
   `sol_work/peaks.json` — so the analyzer skips the per-round
   measured↔SOL correlation, and every roadmap item's
   `expected_gain_pct` rests on a coarser ceiling. Say so up front; if
   the user doesn't want the degraded stage, write
   `sol: {enabled: false}` to skip its wall-clock outright. Never work
   around a missing skill by having an agent recall hardware peaks.
7. **Disaggregated deployment?** A `disagg:` block in `task.yaml` makes the
   workflow drive the checkout's own harness
   (`examples/disaggregated/slurm/benchmark/submit.py`) instead of launching
   `trtllm-serve`, so this host must be able to `sbatch` **and** see the
   cluster paths the harness config names. Everything else about the mode
   is documented on the `disagg` block in `task.example.yaml`.

## 2. Write task.yaml (if the user didn't provide one)

Copy `<agent_flow>/workflows/perf_optimize/task.example.yaml` into the
workspace-to-be and fill it in. Required: `checkpoint_path`,
`trtllm_repo_path`. Ask the user for anything they haven't stated rather
than inventing values:

- `benchmark`: the operating point(s) every measurement replays
  (ISL/OSL/concurrency/num_prompts). Defaults exist, but confirm they
  match the user's deployment shape. `concurrency` is a single int or a
  list of ints — a list turns on Pareto-curve mode: every measurement
  (baseline, each evaluator attempt, each QA round) runs once per point,
  and the evaluator applies the Pareto gate (mean per-point gain must
  pass the thresholds AND no point may regress beyond the noise floor);
  the report gains a Pareto Improvement section (x = tok/s/user,
  y = tok/s/gpu). Benchmark cost multiplies by the point count — keep
  the list short (~3-5 points), or pair it with a `num_prompts` **list**
  (same length, each entry ≥ its point) so low-concurrency points run
  far fewer prompts than high-concurrency ones; that is what makes a
  wide curve affordable.
- **Size the per-measurement cost before launching** — this is the #1
  preflight trap. Estimate one full measurement as
  `Σ_points (num_prompts_i × OSL / expected_agg_tok_s(c_i))` and keep it
  well under the allocation walltime minus ~30 min of env build (the
  campaign replays it for the baseline, every evaluator attempt, and
  the final verification; a measurement that cannot finish inside one
  allocation restarts forever and the campaign never completes).
  Example: 9 points [1..256] × 4096 prompts × OSL 6144 ≈ 30-70 h per
  measurement — infeasible; scaling num_prompts [8..1024] with the
  points brings it to ~1 h.
- `optimize.target_metric` (default `output_throughput`; latency keys
  like `median_ttft_ms`/`median_tpot_ms` also work — gains are
  normalized so positive = better) and optional
  `optimize.target_improvement_pct` (orchestrator-enforced early stop).
- `optimize` budgets — defaults `max_rounds: 5`,
  `max_items_per_round: 3`, `max_attempts_per_item: 3`,
  `accept_fraction: 0.5`, `noise_floor_pct: 1.0` — are sensible. The
  loop **runs the full round budget** (it only ends early when the
  target is met, or an analyzer turn finds no actionable item), so
  `max_rounds × max_items_per_round`
  is what bounds the items a campaign can attempt — 15 with the defaults.
  `optimize.item_execution` defaults to `parallel`; set it to `serial`
  to apply approved items directly in order from the latest campaign state.
  Two things worth telling the user when sizing a run:
  - **Rounds are not equally expensive.** A round pays for a profile
    after an accept, after a reverted code attempt may have changed
    gitignored build output, or when an older checkpoint cannot prove its
    profile is current. Otherwise it opens replan-only — the analyzer
    plans from the standing profile and the round's verdicts without
    touching the GPU. On a config-only plateau `max_rounds` costs almost
    nothing beyond the per-attempt benchmarks; a productive or
    code-mutating campaign pays more profiling turns.
  - The per-item cost driver is the **evaluator benchmark**, one per
    attempt, which no mode avoids. `noise_floor_pct` and `approaches`
    are what keep the item list short.
- `optimize.approaches` (default `[config, code]`): set `[code]` when the
  user wants genuine source-level optimizations only (no tuning-YAML knob
  changes — attempts that touch the tuning file are auto-rejected;
  requires the editable install from preflight step 4), or `[config]`
  when the TRT-LLM checkout must not be modified at all.
- `profile`: `methods` (subset of `[nsys, torch, ncu]`, default all
  three — what a profiling round captures: nsys timeline + torch ops +
  a bounded ncu per-kernel deep dive on the top nsys kernels,
  interpreted with the `perf-nsight-compute-analysis` skill; drop
  entries to trim the cost of the rounds that pay it) and
  `nsys_iter_range` (default `"100-150"`).
- `profile.kernel_coverage`: include (an empty mapping enables the
  defaults `min_share_pct: 0.5`, `coverage_target_pct: 95`) when the
  user wants the **per-kernel coverage contract** — ncu SOL analysis on
  every kernel above the share bar (multi-pass capture) and, per
  kernel, an explicit answer to *can it be made faster?* and *can it be
  fused with its neighbors?* recorded in a schema-validated
  `kernel_ledger.yaml` each profiling round (a roadmap item or an
  evidence-backed dismissal per question; the orchestrator aborts the
  round on an incomplete ledger — replan-only rounds run no ncu and are
  waived — and the report gains a Kernel Coverage accountability section
  resolving every disposition to its outcome). This is the "every kernel
  fusion/optimization possibility considered before done" guarantee; it
  needs `nsys` + `ncu` in `profile.methods` and adds profiling
  wall-clock to every round that profiles.
- `accuracy`: include only if the user has an eval command they want the
  final verification to run at campaign end; omit the block otherwise.
- `extra_llm_api_options`: starting server tuning YAML, if they have one.
  It seeds `<workspace>/tuning/extra_llm_api_options.yaml`, which the
  optimizer then evolves — the workflow always passes the workspace copy
  to `trtllm-serve`.
- `sol`: **the analytical SOL-projection stage is on by default**, so
  omit this block unless the user wants to turn it off or hand the
  skill a hint. Enabled, it produces `sol_projection.md` +
  `sol_work/peaks.json` after the
  baseline; the analyzer ranks roadmap items against the projected
  headroom, correlates each round's per-op measurements against the
  ceiling via the skill's `sol_calc.py analyze` into the findings'
  *SOL correlation* section, and owes a remaining-gap attribution
  whenever it exhausts
  the roadmap with headroom left, the optimizer aims each item's
  realization at the binding ceiling, and the report gains a
  Projection vs Measured headroom-captured section closed by a
  remaining-gap accountability breakdown. Write `sol: {enabled: false}`
  when the user explicitly does not want it — e.g. to save the extra
  stage's wall-clock, or when the SOL skill is unavailable. The
  projector follows the
  `internal-perf-sol-analysis` skill (from the `trtllm-agent-toolkit`
  plugin — an `internal-` skill, so open-source builds strip it;
  install a build that has it for the full methodology). Without it the
  projector falls back to `perf-analysis` and degrades honestly: a
  coarse ceiling grounded on named sources rather than the peaks
  calculator, no peaks file, and no per-round correlation.
  It never recalls a hardware peak. Every field is
  optional — `enabled` gates the stage (default `true`) and `gpu` is
  the part-name hint for the skill's peaks calculator. The stage needs no
  GPU (with local GPUs it additionally measures the skill's latency
  constants) and runs once per campaign — the ceiling is a property of
  hardware + model + operating point, not of the applied optimizations.
  Where a spec or a mapping stays uncertain, the projector is pointed at
  the `internal-glean-search` skill / `internal-glean-specialist`
  subagent as read-only reference, used only if that skill or subagent is
  installed in the session. The hosted-MCP wiring this skill used to
  document (`--glean-mcp-url`, `$PERF_OPTIMIZE_GLEAN_MCP_URL`) is gone: the
  workflow ships from TensorRT-LLM now, and upstream replaced the MCP
  server with that skill. Passing the flag would be a CLI error.

## 3. Launch (long-running — background it)

```bash
perf-optimize --task <path>/task.yaml --workspace workspace/perf-optimize/<model-name> \
    > <somewhere>/perf_optimize.log 2>&1
```

A campaign runs for hours (each round: a full benchmark per attempt + a
profiled replay per accept, plus one re-profile when the standing runtime
evidence became stale; plus one final-verification benchmark at the end). Launch it in the background
(`run_in_background` / `nohup`) with output captured to a log file, then
monitor.

- **Resume**: re-running the identical command resumes from
  `<workspace>/.perf_optimize_state.json` — interruption (Ctrl-C, crash,
  node loss) is safe.
- **Fresh start**: `--clean` wipes the workspace's managed state but
  never touches the TRT-LLM checkout (abandoned `perf-optimize/*`
  branches are left for inspection).
- `--max-rounds N` overrides the round budget on a fresh run only;
  ignored on resume.
- `--reuse-analysis <dir>` starts a fresh run **at the optimize stage**
  by importing a previous `perf-analyze` workspace or `perf-optimize`
  campaign workspace: its baseline report (+ result JSONs), SOL
  projection (+ `sol_work/`), and newest profile findings (+ traces,
  `kernel_ledger.yaml`) are copied in, the benchmarker/projector are
  skipped, and round 1's analyzer plans from the imported evidence
  without launching a server or a profiler. Rounds 2+ profile normally.
  Reach for this when the user has *just* run `perf-analyze` on the same
  deployment, or is starting a follow-up campaign on a machine whose
  baseline has not changed — it saves the two most expensive stages.
  Check before proposing it: the imported baseline is what every gain is
  measured against, so the source run must describe the **same** model,
  checkpoint, hardware, parallel mapping and operating point. When in
  doubt, don't reuse the baseline (the import is per-artifact — point at
  a source with only findings, or just run normally). A source
  `roadmap.yaml` is kept as read-only prior art in
  `reused_analysis/prior_roadmap.yaml`, never as this campaign's ledger;
  `reused_analysis/manifest.md` records the provenance and the report
  repeats it. Fresh runs only — ignored on resume.

## 4. Monitor

Poll the workspace (and the launch log) rather than waiting silently:

- `progress.yaml` — append-only audit log; new entries mean it's alive.
- `roadmap.yaml` — the ranked plan; watch item `status` and measured
  gains vs `expected_gain_pct`; `current_best` tracks the last accepted
  measurement.
- `baseline/benchmark_results.md` (and `sol_projection.md` right after
  it unless `sol.enabled: false`), then per-round
  `rounds/round_<n>/` (`analysis/profile_findings.md`,
  `item_<j>_<id>/attempt_<k>/` — accepted attempts also grow a
  `profile/` nsys capture), and at the end
  `final_verification/verification_report.md`. A round whose standing
  runtime profile is still current writes a short replan note in place
  of a profiling report, and no traces — that is the replan-only mode,
  not a stalled analyzer.
- `git -C <trtllm_repo_path> log --oneline` on the `perf-optimize/*`
  branch — one commit per accepted item.

If it dies, read the tail of the launch log, fix the environment issue,
and re-run the same command to resume.

## 5. Wrap up

When the run finishes (round budget spent, roadmap exhausted, or the
improvement target met), report to the user from
`optimization_report.md` / `.html`:

- the verified cumulative improvement vs baseline on the target metric
  (from the final verification's independent benchmark),
- the optimization trajectory (baseline → each accepted item → final
  verification; the HTML renders it as a line chart),
- expected-vs-measured per applied item, failed attempts with reasons,
- the kernel-level before/after comparison — the "after" side comes from
  the **last accepted attempt's `profile/`** (the accept-evidence nsys
  capture the evaluator takes on every APPROVE; rejects are reverted, so
  it always reflects the final accepted state), else the last round's
  profile,
- final config (`tuning/extra_llm_api_options.accepted.yaml`) and the
  code diff on the `perf-optimize/*` branch,
- when the projector ran: the Projection vs Measured section — baseline
  vs final % of SOL, i.e. how much of the analytically projected
  headroom the campaign captured, which bound class the remaining gap
  sits in, and the remaining-gap accountability breakdown (every part
  of the gap `closed` / `infeasible: <constraint>` / `untried` /
  `unexplained`, each verdict citing evidence — relay this to the user:
  it is the answer to "why does the campaign end here?"),
- remaining roadmap items as future work.

Point the user at the branch + accepted config for productionizing
(cherry-pick / PR is theirs to drive — the workflow never pushes).

## Pitfalls

A Slurm allocation is time-limited; when its walltime expires the node
is taken back, which also kills the agent-flow process. When that
happens, allocate a fresh node and launch the perf-optimize workflow
again with the same command — it resumes from `<workspace>/.perf_optimize_state.json`
at the interrupted stage.

The walltime ceiling is usually enforced by the *partition*, not by the
QoS list your account carries: a partition can set `DenyQos=...`, so a
long QoS that `sacctmgr show assoc` happily lists is still rejected at
submit with "Invalid qos specification" and only a short QoS passes.
Read the partition's own limits (`scontrol show partition <partition>`)
rather than your association, and size the benchmark block to the
window you actually get (see the per-measurement cost note above).

For unattended multi-window campaigns, don't rely on the driving
session staying alive to resubmit: run a small nohup'd **keeper loop**
on the launch host that every ~5 min resubmits the sbatch iff the
workflow state JSON has `done: false` and the queue has no job of that
name (double-check an empty queue reading ~60 s apart before
resubmitting; guard with a pidfile and a stop-sentinel file). A
session restart otherwise turns a walltime kill into a silent
hours-long stall.

## Improvement suggestions

This workflow and this skill are under active development. If driving
a run surfaces an issue — a bug or crash in agent-flow, a misleading
log or report, a preflight check this skill is missing, a stale or
wrong instruction — don't just work around it silently. Note it while
it's concrete, and when reporting results to the user, include a short
list of workflow improvement suggestions: what went wrong, where
(file / step), and the fix you'd propose. If a fix is a small edit to
this skill or the workflow docs, offer to apply it.
