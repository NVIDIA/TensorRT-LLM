# perf-analyze

A linear pipeline that serves a model with `trtllm-serve`,
benchmarks and profiles it with TensorRT-LLM's
`tensorrt_llm/serve/scripts/benchmark_serving.py`, and writes a report
whose headline is the **main performance bottleneck**.

All roles run on the **Claude Code** backend:

```
benchmarker ──▶ projector ──▶ analyzer ──▶ reporter
```

- **Benchmarker** — launches `trtllm-serve`, drives the configured
  benchmark operating point(s) (one ISL/OSL/concurrency; a
  `concurrency` list means one run per point over the same server —
  Pareto-curve mode), records the clean latency/throughput numbers in
  `benchmark_results.md`.
- **Projector** *(on by default — skipped only when `task.yaml` sets
  `sol.enabled: false`)* — derives a first-principles **speed-of-light
  (SOL) ceiling**
  at the measured operating point, following the
  **`internal-perf-sol-analysis` skill** (from the `trtllm-agent-toolkit`
  plugin) as its methodology: hardware peaks come from the skill's peaks
  calculator (never from memory), latency constants from its
  `measure_channels.py` when a GPU is reachable, and the ceiling from
  the skill's α-β-u model (which also prices launch-bound regimes).
  It writes `sol_projection.md` — the projected SOL
  ceiling for TTFT/TPOT/throughput, the **% of SOL** headline with
  measured MFU/MBU, and the bound breakdown
  (compute / memory / launch) that says where the headroom lives — and
  persists the machine-readable `sol_work/peaks.json` (latency
  constants merged in) that the analyzer's correlation joins against.
- **Analyzer** — replays the same load under **Nsight Systems (nsys)**,
  the **PyTorch profiler**, and **Nsight Compute (ncu)** — the ncu run
  is a bounded per-kernel deep dive on the top nsys kernels, captured
  over the same iteration window and interpreted with the
  **`perf-nsight-compute-analysis` skill** (per-kernel SOL%, occupancy,
  warp stalls → bound class) — mines the traces, and writes ranked
  bottleneck hypotheses to `profile_findings.md`, each hypothesis
  synthesized across the analyses (nsys timeline / ncu kernel analysis /
  SOL correlation). When the projector ran, it also
  runs the skill's **measured↔SOL correlation** (`sol_calc.py analyze`):
  it rolls the traces up into per-op regions (`sol_work/regions.json`),
  joins them against the projector's `sol_work/peaks.json`, and reports
  the joined per-op table (% of SOL, MFU/MBU, gap, bound per region) in
  the findings' *SOL correlation* section. This is the same diagnosis
  role as perf-optimize's analyzer — perf-optimize adds roadmap
  authoring on top — so perf-optimize is an extension of this pipeline.
- **Reporter** — synthesizes everything into `performance_report.md` (+ a
  self-contained `performance_report.html` that renders the top-kernel
  table as an inline-SVG share-bar chart — no CDN, chart omitted when
  nsys produced no table), classifying the single dominant bottleneck
  and recommending fixes — every recommendation grounded in the three
  analyses the findings carry (nsys timeline, ncu kernel analysis, SOL
  correlation/projection when present), naming which support it. When
  the projector ran, the report carries a
  **Projection vs Measured** section and the verdict/recommendations must
  state how the projected headroom was weighed.

The pipeline is **one-shot** (no review loop): each stage runs once and
checkpoints before advancing, so an interrupted run resumes at the same
stage.

## Usage

```bash
perf-analyze \
    --task path/to/task.yaml \
    --workspace workspace/perf-analyze/<model-name>
```

- Re-running the same command **resumes** from the checkpoint
  (`.perf_analyze_state.json`) when one is present.
- `--clean` wipes the checkpoint and managed output files and starts fresh.

Copy [`task.example.yaml`](./task.example.yaml) and fill it in.

## `task.yaml`

| Field | Required | Notes |
| --- | --- | --- |
| `checkpoint_path` | ✅ | Model checkpoint dir to serve (must exist). |
| `trtllm_repo_path` | ✅ | TensorRT-LLM checkout providing `trtllm-serve` + `benchmark_serving.py` (must exist). |
| `extra_llm_api_options` | optional | Path to a YAML passed verbatim to `trtllm-serve --extra_llm_api_options` — the single place for all server tuning (parallelism, batch sizes, KV-cache fraction, CUDA-graph config, ...). Omit to use server defaults. The server always runs the `pytorch` backend on `127.0.0.1:8000`. |
| `benchmark` | optional | Operating point: `dataset_name`, `random_input_len` (ISL), `random_output_len` (OSL), `num_prompts`, `concurrency`, `request_rate`, `dataset_path`. `concurrency` is a positive int (single operating point, default `64`) **or a non-empty list of them** — a list turns on **Pareto-curve mode**: the benchmarker runs `benchmark_serving.py` once per point (sorted ascending, deduplicated, over one server launch), the analyzer profiles at the largest point, and the report gains a measured Pareto curve (x = tok/s/user = `1000/mean_tpot_ms`, y = tok/s/gpu = `output_throughput/num_gpus`) with its own chart in the HTML companion. `num_prompts` is a positive int (same count at every point, default `200`) or — curve mode only — a list paired index-by-index with the `concurrency` list (each entry ≥ its point; sorted together with it), so low-concurrency points can run far fewer prompts. |
| `profile` | optional | `methods` (subset of `[nsys, torch, ncu]`, default all three) and `nsys_iter_range` (default `"100-150"`, gating nsys/torch server-side and the ncu capture via `--profile-from-start off`). |
| `slurm-environment` | optional | When present (`slurm_partition` + `docker_image`), the server + benchmark run inside a Slurm-launched container instead of locally. |
| `sol` | optional | Gates the projector stage (the SOL ceiling, per the `internal-perf-sol-analysis` skill), which runs **by default**. Every field is optional — `enabled` is the gate (default `true`) and `gpu` is the part-name hint for the skill's peaks calculator when the automatic mapping would guess wrong. Omit the block entirely to accept the defaults. |

The spec is validated at the CLI boundary; defaults for the optional
blocks are resolved and the normalized spec is written to
`<workspace>/task.yaml` so the agents read fully-explicit values.

> **Renamed field:** `benchmark.max_concurrency` is now
> `benchmark.concurrency` — the old name fails validation with a
> pointer to the new one. Task files written before the rename (and
> resumed pre-rename workspaces) need the one-line edit.

## Workspace layout

```
workspace/perf-analyze/<name>/
├── .perf_analyze_state.json   # checkpoint
├── task.yaml                         # resolved spec (defaults filled in)
├── serve.log, serve.pid              # server run artifacts
├── benchmark_results.md              # ← benchmarker
├── <backend>-...json                 # raw benchmark_serving result
├── sol_projection.md                 # ← projector (blank when disabled)
├── sol_work/                         # ← projector peaks.json; analyzer regions.json + sol.json
├── server_nsys.nsys-rep, nsys_stats.txt   # ← analyzer (nsys)
├── torch_trace/, perf_metrics.json        # ← analyzer (torch)
├── server_ncu.ncu-rep, ncu_details.txt, ncu_raw.csv   # ← analyzer (ncu)
├── profile_findings.md               # ← analyzer
├── performance_report.md / .html     # ← reporter (the deliverable)
└── progress.yaml                     # append-only audit log
```

## Notes

- **Local vs Slurm.** Without a `slurm-environment` block, the agents run
  `trtllm-serve` and `benchmark_serving.py` directly via `Bash` on the
  current node's GPUs. With it, they run inside a Slurm-launched container.
- **Profiling knobs are verified at runtime.** Env-var names
  (`TLLM_PROFILE_START_STOP`, the torch-profiler dir var) differ across
  TensorRT-LLM versions, so the Analyzer greps the checkout in
  `trtllm_repo_path` to confirm the actual names before relying on them.
- The command knowledge lives in the prompts, so the workflow does not
  depend on any optional agent-toolkit skills being installed. The
  Benchmarker and Analyzer drive `benchmark_serving.py`, `nsys profile`,
  and `ncu` from a **canonical command template** in their system
  prompts (fixed flags — `--trust-remote-code`, `--random-ids`,
  `--tokenize-on-client`, `--ignore-eos`, `--no-test-input`; for nsys
  `-t 'cuda,nvtx,python-gil'`,
  `--cuda-graph-trace node`, `TLLM_NVTX_DEBUG=1`,
  `--trace-fork-before-exec=true`; and for ncu
  `--target-processes all`, `--profile-from-start off`, a bounded
  `--launch-count`, and a `--kernel-name` filter built from the top
  nsys kernels) rather than improvising per run; only
  the paths and the `benchmark` / `profile` values are filled in.
- **ncu kernel analysis (Run C).** The Analyzer's ncu pass answers *why*
  the hot kernels are slow, complementing nsys's *where the time goes*:
  per-kernel Compute/Memory SOL%, occupancy, and warp stalls, classified
  with the `perf-nsight-compute-analysis` skill's thresholds
  (compute- / memory- / latency-bound). It degrades gracefully — a
  missing `ncu` binary or profiling permission (`ERR_NVGPUCTRPERM`)
  skips the run with an honest `ncu unavailable` line in the findings —
  and its replayed benchmark numbers are never reported as measurements
  (kernel replay serializes the GPU). On multi-rank serves the coverage
  is partial by construction: replaying a CUDA-graph kernel under MPI
  costs on the order of a minute per launch, and TensorRT-LLM's
  executor hang watchdog (`_torch/pyexecutor/hang_detector.py`, fixed
  300 s, no config/env knob) kills the server after a handful of
  profiled launches — ncu writes the report incrementally, so the
  analyzer keeps what was captured, may add small targeted passes, and
  states the achieved share of GPU time in the findings.
- **Optimization casebook (optional reference).** The Benchmarker and
  Analyzer are directed to load the `perf-optimization-casebook` skill
  (from the `trtllm-agent-toolkit` plugin) early, as **read-only**
  reference, so their analysis is anchored to known TRT-LLM performance
  precedents (its *bottleneck signal → candidate pattern* index) — the
  Analyzer tags each ranked hypothesis with the pattern it matches. This is
  reference-only: the roles never apply optimizations from it, and if the
  skill is not installed they note that and proceed, so it stays an
  enhancement rather than a hard dependency.
- **SOL projection (default-on stage).** The Projector runs unless
  `task.yaml` sets `sol.enabled: false`. Its methodology is
  the `internal-perf-sol-analysis` skill (loaded via the `Skill` tool):
  hardware peaks come from the skill's peaks calculator (never quoted
  from memory), latency constants
  from its `measure_channels.py` **when a GPU is reachable** (local
  runs — between stages the GPUs are idle; under a `slurm-environment`
  block the projector runs on the login node and records the α terms
  as unmeasured instead of guessing), and the ceiling from the skill's
  α-β-u model, with the arithmetic recorded in the report so every
  number can be re-checked. The skill's bundled scripts are the only
  thing the projector executes. That skill is `internal-` prefixed, so
  **open-source builds of the toolkit strip it** while keeping
  `perf-analysis`. Which of the two this session has is resolved **in
  Python** before the run (`sol_methodology.resolve_sol_methodology`,
  one ~1 s probe of the live skill list — a session connection, no model
  call — skipped entirely when the stage is off), so the projector is
  told to load a skill that is actually there; an unreachable probe
  **fails open** to the SOL skill, because silently downgrading a stage
  the user asked for is worse than a wasted check. Without it the
  projector loads `perf-analysis` instead and works the same methodology
  without a calculator: the peaks come from named sources, marked as not
  calculator-resolved, and no `sol_work/peaks.json` is written. When
  no defensible ceiling can be grounded at all, it writes an
  honest "Projection unavailable: <reason>" `sol_projection.md` and the
  pipeline continues on measured evidence alone — projected numbers are
  never fabricated. With the projection present, the Analyzer loads the
  same skill and correlates its fresh profile against the ceiling
  (`sol_calc.py analyze` over `sol_work/regions.json` + the projector's
  `sol_work/peaks.json`, writing `sol_work/sol.json`) — structural facts
  only, degrading to an honest "Correlation unavailable" line when a
  precondition fails (a missing peaks file included). Where a spec or
  mapping stays uncertain, the Projector is pointed at the
  `internal-glean-search` skill / `internal-glean-specialist` subagent
  as **read-only reference**, used only if that skill/subagent is
  installed in the session.
- **Resume with the same task file.** Stage gating on resume reads the
  workspace's checkpointed `task.yaml`, while prompt augmentation reads
  the `--task` file — pass the same task file when resuming so the two
  cannot disagree about the `sol` / `slurm-environment` blocks.
