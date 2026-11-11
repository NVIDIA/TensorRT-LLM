---
id: case-cudagraph-default-batch-sizes
type: regression-case
family: execution-and-graph
module: cuda-graph-and-compile
maturity: full
regression_class: [cuda-graph-regression]
signals: [perf-ci-bar-failure, throughput-drop]
subsystems: [cuda-graph, perf-test-config]
introduced_via: [config-default-change]
phase: [decode]
patterns: [pattern-default-change-regressed-tuned-workload]
nvbugs: ["6115290"]
commits: ["84c69deaa070"]
success_prs: [13743]
failed_prs: []
---

# Changed default CUDA-graph batch-size list regressed a tuned GB200 workload

> Part of the [CUDA graph & compile regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6115290` · commit `84c69deaa070` · PR #13743 —
  "[fix] Fix GPT OSS 120B GB200 Test Regression".
- **Symptom:** the GPT OSS 120B FP4 perf-sanity test on GB200
  (`gpt_oss_fp4_dep4_1k8k-con2560_iter5_1k8k`, config
  `gpt_oss_120b_fp4_grace_blackwell.yaml`, `max_batch_size: 640`,
  padding-enabled CUDA graphs) regressed its metric after a default change;
  surfaced via the perf-sanity CI bar. The fix PR states no numbers, only
  that "the test now has the same metric as before".
- **Root cause:** PR #12895 changed the *default* padding-enabled CUDA-graph
  capture list in `CudaGraphConfig` (`tensorrt_llm/llmapi/llm_args.py`) from
  "multiples of 8 up to 128, then powers of two up to `max_batch_size`" to a
  "+64 stride" extension above 128. The workload's metric baseline was
  established under the old implicit capture set (the fix restores that exact
  list; per the PR, "the test now has the same metric as before"). The new
  denser set was slower because the +64 stride introduces non-power-of-2
  capture sizes above 128 (192, 320, 384, 448, 576); padding a batch to one
  of those rather than to a power of two changes the shape the downstream
  GEMM, MoE all-to-all and attention kernels run at. So the padded graph
  itself is fine — the shape it pads *to* is what the downstream kernels are
  tuned for.
- **How introduced:** commit `7e5275fc9175` / PR #12895 — "[perf] Use +64
  batch sizes for padding-enabled CUDA graphs" — a global default change
  intended as a perf improvement (per its `[perf]` tag) that regressed this
  workload (named in the fix PR).
- **Fix mechanism:** pin the original expansion explicitly in the test config:
  `cuda_graph_config.batch_sizes: [1, 2, 4, 8, ..., 128, 256, 512, 640]` in
  `tests/scripts/perf-sanity/aggregated/gpt_oss_120b_fp4_grace_blackwell.yaml`,
  so this workload no longer depends on the default.
- **Detection signal:** perf bar trips right after a commit touching default
  CUDA-graph batch-size generation while the workload's YAML leaves
  `batch_sizes` unset; check
  `git log -S "batch_sizes" -- tensorrt_llm/llmapi/llm_args.py` over the
  regression window, and diff the effective (logged) capture list against the
  last-good build — specifically for non-power-of-2 sizes above 128, then
  compare per-kernel times at those batch sizes (MoE all-to-all and the
  swiGlu GEMM first).
- **Prevention/guard:** none added — the fix pins one test config only. Gap:
  changes to `CudaGraphConfig` default batch-size generation should be swept
  across the perf-sanity workload matrix before merge, and tuned perf-sanity
  configs should pin `batch_sizes` explicitly rather than inherit defaults.
- **Generalizes to:** pattern-default-change-regressed-tuned-workload — a
  "better on average" default silently retunes every workload that inherited
  the old default. Carries to: scheduler/admission defaults (e.g.
  `max_num_tokens` heuristics), KV-cache defaults such as
  `free_gpu_memory_fraction`, autotuner default candidate sets, and any
  default kernel-backend flip where perf configs relied on the prior choice.
