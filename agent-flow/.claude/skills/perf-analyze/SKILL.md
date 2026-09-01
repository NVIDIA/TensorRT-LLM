---
name: perf-analyze
description: Launch and operate this repo's perf-analyze workflow, which DIAGNOSES a TensorRT-LLM serving deployment without applying changes — benchmark at one concurrency or a Pareto curve of them (tok/s/user vs tok/s/gpu), analytical SOL projection on by default (via the internal-perf-sol-analysis skill), nsys + torch-profiler + ncu per-kernel deep dive (via the perf-nsight-compute-analysis skill), and a report naming the single dominant bottleneck. Use when the user wants to analyze / profile / diagnose trtllm-serve performance (throughput, TTFT, TPOT, ITL, e2e latency) or says "run perf-analyze". To actually APPLY optimizations, use the perf-optimize workflow instead.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Operating the perf-analyze workflow

`perf-analyze` is this repo's one-shot diagnosis pipeline for
`trtllm-serve`: benchmarker → projector → analyzer → reporter. It
serves the model, benchmarks one operating point, derives an analytical
speed-of-light (SOL) ceiling (via the `internal-perf-sol-analysis`
skill — the projector runs unless `sol.enabled: false` turns it off),
profiles the same load under nsys + the torch profiler + a bounded ncu
per-kernel deep dive on the top nsys kernels (the analyzer — the same
diagnosis stage perf-optimize runs; the ncu pass follows the
`perf-nsight-compute-analysis` skill, and with the projector on the
analyzer also correlates the measured per-op times against the ceiling
via the SOL skill's `sol_calc.py analyze`), and writes
`<workspace>/performance_report.md` (+ `.html`) whose headline is the
**single dominant bottleneck**. It never applies optimizations and
never mutates the TRT-LLM checkout.

Do not hand-roll a serve/benchmark/profile loop when the user asks to
diagnose serving performance — drive this workflow instead.
Authoritative references (read them before answering detailed
questions):

- `<agent_flow>/workflows/perf_analyze/README.md` — pipeline contract,
  task schema, workspace layout, resume semantics.
- `<agent_flow>/workflows/perf_analyze/task.example.yaml` — fully
  commented task template.

The workflow itself now ships from TensorRT-LLM, under `agent-flow/`;
the paths below are relative to the INSTALLED `agent_flow` package. Print
its location with
`python -c "import agent_flow,pathlib;print(pathlib.Path(agent_flow.__file__).parent)"`,
or read them out of `<trtllm_repo_path>/agent-flow/agent_flow/`.

Use perf-analyze when the user only wants a bottleneck diagnosis.
Use `perf-optimize` (same repo) when they want changes applied and
verified.

## 1. Preflight — verify before launching

Run these checks; fix or ask only where noted.

1. **CLI installed**: `perf-analyze --help` works. If not:
   `pip install -e <trtllm_repo_path>/agent-flow`. The workflow drives
   the Claude Code backend, so the `claude` CLI must be installed and
   signed in; `CLAUDE_CODE_DEFAULT_MODEL` overrides the model if the
   user wants.
2. **Paths from the user**: `checkpoint_path` (model checkpoint dir) and
   `trtllm_repo_path` (TensorRT-LLM checkout) must exist — the CLI
   refuses to start otherwise. The same applies to
   `extra_llm_api_options` when set.
3. **No git requirements**: unlike perf-optimize, this workflow is
   read-only with respect to the TRT-LLM checkout — no branch, no
   commits, no reverts. A dirty worktree is fine.
4. **Install matches checkout**: the server runs the *installed*
   `tensorrt_llm`, while `benchmark_serving.py` and the profiling
   env-var names are read from `trtllm_repo_path`. They should be the
   same version — `python -c "import tensorrt_llm; print(tensorrt_llm.__file__)"`
   resolving into `trtllm_repo_path` (editable install) is the clean
   setup. A mismatch won't stop the run but can skew the analyzer's
   knob-verification grep; warn the user.
5. **Environment**: either the current node has GPUs (`nvidia-smi`
   works), or the task carries a `slurm-environment` block that routes
   the server + benchmark through a Slurm-launched container. On a
   login or head node with no local GPUs, bring up the node and
   container with your own site's recipe first. When
   `profile.methods` includes `nsys` (the default), `nsys` must be on
   PATH where the server runs — it is in the dev container;
   likewise `ncu` for the `ncu` method (also in the dev container — a
   missing binary or profiling permission degrades that run gracefully;
   on multi-rank serves expect *partial* per-kernel coverage — TRT-LLM's
   fixed 300 s executor hang-watchdog kills the server after a few
   replayed CUDA-graph launches, and the findings state the achieved
   coverage).
6. **Internal toolkit skills (affects depth, not launch)**: the
   default-on SOL projector wants `internal-perf-sol-analysis`. That is
   an `internal-`prefixed skill, so **open-source builds of the
   `trtllm-agent-toolkit` plugin strip it** while keeping
   `perf-analysis`. **The CLI checks this itself** at launch and prints
   one line when the skill is missing, so this is context for the user
   rather than something you must gate on. The run completes either way:
   without the SOL skill the projector falls back to `perf-analysis`,
   grounds the peaks from named sources rather than the skill's
   calculator, marks them as such, and writes no `sol_work/peaks.json` —
   so the analyzer skips the measured↔SOL correlation. Say so up front;
   if the user doesn't want the degraded stage, write
   `sol: {enabled: false}` to skip its wall-clock outright. Never work
   around a missing skill by having an agent recall hardware peaks.

## 2. Write task.yaml (if the user didn't provide one)

Copy `<agent_flow>/workflows/perf_analyze/task.example.yaml` into the
workspace-to-be and fill it in. Required: `checkpoint_path`,
`trtllm_repo_path`. Ask the user for anything they haven't stated rather
than inventing values:

- `benchmark`: the operating point(s). Defaults: `dataset_name: random`,
  ISL 1024 / OSL 128, `num_prompts: 200`, `concurrency: 64`; optional
  `request_rate` and `dataset_path`. `concurrency` is a single int
  (one operating point) **or a list of ints** — a list turns on
  Pareto-curve mode: one benchmark run per point over the same server,
  profiling at the largest point, and the report gains a measured
  Pareto curve (x = tok/s/user = 1000/mean_tpot_ms, y = tok/s/gpu =
  output_throughput/num_gpus). Benchmark time scales with the point
  count; in curve mode `num_prompts` may also be a **list** paired
  index-by-index with the concurrency list (each entry ≥ its point) so
  low-concurrency points run far fewer prompts — estimate
  `Σ num_prompts_i × OSL / agg_tok_s(c_i)` and keep it well inside the
  allocation walltime. Confirm the point(s) match the user's deployment
  shape — the whole diagnosis is anchored to them.
- `extra_llm_api_options`: server tuning YAML passed verbatim to
  `trtllm-serve --extra_llm_api_options` — the single place for all
  server knobs (parallelism, batch sizes, KV-cache fraction,
  CUDA-graph config, ...). Omit for server defaults. The server always
  runs the `pytorch` backend on `127.0.0.1:8000`.
- `profile`: `methods` (subset of `[nsys, torch, ncu]`, default all
  three) and `nsys_iter_range` (default `"100-150"`, the steady-state
  iteration window `TLLM_PROFILE_START_STOP` captures; the ncu deep
  dive arms on the same window via `--profile-from-start off`).
- `slurm-environment`: include only when the server + benchmark must
  run inside a Slurm-launched container; both `slurm_partition` and
  `docker_image` are then required.
- `sol`: **the analytical SOL-projection stage is on by default**, so
  omit this block unless the user wants to turn it off or hand the
  skill a hint. Enabled, it produces `sol_projection.md` +
  `sol_work/peaks.json`, the analyzer's measured↔SOL per-op correlation
  in `profile_findings.md`, and a Projection vs Measured section in the
  report. Write `sol: {enabled: false}` when the user explicitly does
  not want it — e.g. to save the extra stage's wall-clock, or when the
  SOL skill is unavailable. The
  projector follows the `internal-perf-sol-analysis` skill (from the
  `trtllm-agent-toolkit` plugin — an `internal-` skill, so open-source
  builds strip it; install a build that has it for the full
  methodology). Without it the projector falls back to `perf-analysis`
  and degrades honestly: a coarse ceiling grounded on named sources
  rather than the peaks calculator, no peaks file, and no measured↔SOL
  correlation downstream. It never recalls a hardware peak.
  Every field is optional — `enabled` gates the stage (default `true`)
  and `gpu` is the part-name hint for the skill's peaks calculator. The stage needs no
  GPU (with local GPUs it additionally measures the skill's latency
  constants).
  Where a spec or a mapping stays uncertain, the projector is pointed at
  the `internal-glean-search` skill / `internal-glean-specialist`
  subagent as read-only reference, used only if that skill or subagent is
  installed in the session. The hosted-MCP wiring this skill used to
  document (`--glean-mcp-url`, `$PERF_ANALYZE_GLEAN_MCP_URL`) is gone: the
  workflow ships from TensorRT-LLM now, and upstream replaced the MCP
  server with that skill. Passing the flag would be a CLI error.

## 3. Launch (long-running — background it)

```bash
perf-analyze --task <path>/task.yaml --workspace workspace/perf-analyze/<model-name> \
    > <somewhere>/perf_analyze.log 2>&1
```

A run takes on the order of one to a few hours (server spin-up + one
benchmark, plus a profiled replay per profiling method). Launch it in
the background (`run_in_background` / `nohup`) with output captured to
a log file, then monitor.

- **Resume**: re-running the identical command resumes from
  `<workspace>/.perf_analyze_state.json` at the stage that was
  interrupted — Ctrl-C, crash, node loss are all safe. Pass the **same
  `--task` file** when resuming: stage gating reads the checkpointed
  workspace `task.yaml` while prompt selection reads the `--task` file,
  and they must agree about the `sol` / `slurm-environment` blocks.
- **Fresh start**: `--clean` wipes the checkpoint and managed outputs
  (`benchmark_results.md`, `sol_projection.md`,
  `profile_findings.md`, `performance_report.md/.html`,
  `progress.yaml`); run artifacts (`serve.log`, result JSON,
  `*.nsys-rep`, `torch_trace/`) are left alone.
- A workspace holding non-empty outputs but **no checkpoint** refuses
  to start (`FileExistsError`) — pass `--clean` or pick a new
  workspace directory.

## 4. Monitor

Poll the workspace (and the launch log) rather than waiting silently:

- `progress.yaml` — append-only audit log; new entries mean it's alive.
- Stage deliverables appear in order: `benchmark_results.md` →
  `sol_projection.md` (unless `sol.enabled: false`) →
  `profile_findings.md` → `performance_report.md` / `.html`.
- Run artifacts: `serve.log` (server health), the raw benchmark result
  JSON, `server_nsys.nsys-rep` + `nsys_stats.txt`, `torch_trace/`,
  `server_ncu.ncu-rep` + `ncu_details.txt` / `ncu_raw.csv`,
  `perf_metrics.json`.

If it dies, read the tail of the launch log. A
`RuntimeError: ... left required output empty/missing` means a stage
ended its turn without writing its deliverable — the checkpoint is left
un-advanced, so re-running the same command retries that stage. Fix any
environment issue and re-run.

## 5. Wrap up

When the run finishes (`✔ performance report written`), report to the
user from `performance_report.md` / `.html`:

- the **single dominant bottleneck** (the report names exactly one
  headline category) and the key trace evidence behind it,
- the headline benchmark metrics at the configured operating point,
- **Projection vs Measured** and the projected headroom, when the
  projector ran (or note the projection declared itself unavailable),
- the top recommendations — stressing that nothing was applied; this
  workflow is diagnosis-only,
- point them at `performance_report.html` (self-contained, renders the
  top-kernel share-bar chart) and the raw traces for their own digging.

If the user wants the recommendations acted on, offer to run the
`perf-optimize` workflow next — its roadmap can start from this report.

## Pitfalls

A Slurm allocation is time-limited; when its walltime expires the node
is taken back, which also kills the agent-flow process. When that
happens, allocate a fresh node and launch the perf-analyze workflow
again with the same command — it resumes from `<workspace>/.perf_analyze_state.json`
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
