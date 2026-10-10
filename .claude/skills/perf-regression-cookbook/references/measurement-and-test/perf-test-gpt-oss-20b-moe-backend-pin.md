---
id: case-perf-test-gpt-oss-20b-moe-backend-pin
type: regression-case
family: measurement-and-test
module: measurement-and-test
maturity: full
regression_class: [measurement-artifact, kernel-selection-regression]
signals: [perf-ci-bar-failure, throughput-drop, slower-kernel-in-trace]
subsystems: [perf-test-config, moe]
introduced_via: [prior-fix-side-effect]
phase: [any-phase]
patterns: [pattern-measurement-not-product]
nvbugs: ["6175923", "6144334"]
commits: ["c61705ec6b71"]
success_prs: [14612]
failed_prs: []
---

# An over-broad perf-test pattern pinned GPT-OSS 20B to the TRITON MoE backend on Blackwell

> Part of the [Measurement & test regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6175923` · commit `c61705ec6b71` · PR #14612 —
  "[https://nvbugs/6175923][test] Revert gpt_oss_20b perf MoE-backend pin";
  the PR states it fixes "NVBugs 6175923 (and duplicate 6144334)", and
  6144334 is marked as a duplicate of 6175923.
  **Do not extend that dup chain mechanically:** 6175923 is in turn marked as
  a duplicate of 6176224, and 6176224 is the *nemotron* defect fixed by
  PR #14003 — a different root cause. Following the duplicate link one more
  hop would fold two unrelated defects into this case. (Read both duplicate
  links from the full bug record; a summary view can show them blank for all
  three of these bugs.)
  **Both bugs are multi-model, and this case fixes only the gpt_oss_20b
  half.** 6175923 reports a 14 %–376 % regression on gpt_oss_20b and
  nemotron_nano_12b_v2, and 6144334 an Inference_Time regression on
  nemotron_nano and gpt_oss models — one QA sweep (1.3.0rc13 `b9ce4b69` →
  1.3.0rc14 `93cb6518`) filed one bug per *sweep*, spanning two models with
  **two unrelated root causes and two different fix PRs**. The
  `nemotron_nano_12b_v2` half is the same regression as nvbug `6176224`,
  root-caused to per-slot D2H syncs in the mamba-hybrid cache manager and
  fixed by **PR #14003** — see
  [mamba-hybrid recurrent-state D2H syncs](../kv-cache-manager/mamba-hybrid-recurrent-state-d2h-syncs.md).
  6175923 marks its gpt_oss_20b rows as removed after #14612 landed, leaving
  the nemotron rows behind — which is why 6175923 was closed without
  verification while 6176224 was closed as verified.
  **Lesson for triage:** a QA-sweep bug whose title names two models is not
  one defect. Never accept a single fix PR as closing it without checking
  every model row; conversely, finding "the" fix PR for such a bug id tells
  you nothing about the other rows.
- **Symptom:** the `gpt_oss_20b_fp4-bench-pytorch-float4` perf tests
  regressed between 1.3.0rc13 (`b9ce4b69`) and 1.3.0rc14 (`93cb6518`) —
  14 %–376 % on Inference Time / Seq Throughput, on B200, per nvbug 6175923;
  surfaced via the QA perf sweep. Per PR #14612 the product had not
  regressed — the test was configured onto a slower MoE backend.
- **Root cause:** PR #12796 added a pattern block to
  `tests/integration/defs/perf/pytorch_model_config.py` whose `patterns`
  list — the bare string `gpt_oss_20b_fp4-bench-pytorch-float4` — matched
  *every* test name carrying that model label, and pinned
  `moe_config: {backend: 'TRITON'}` (alongside `enable_chunked_prefill:
  False`, `enable_attention_dp: False`, a `cuda_graph_config`, a
  `kv_cache_config` and `print_iter_log`). The same model label is also
  exercised on Blackwell via `llm_perf_sanity.yml`, where the AUTO
  resolver in `ModelConfig.resolve_moe_backend` picks `TRTLLM` as the
  optimal backend. Forcing `TRITON` there ran the slower path.
- **How introduced:** a prior fix's side effect, scoped to the wrong
  breadth. The removed block's own comment names its motivation — "GPT-OSS
  20B (NVBug 5720470: MMHA vs XQA kernel regression)" — and it arrived
  with commit `a76136803605`, PR #12796 "[None][test] add unit test and
  e2e test for gpt_oss_20b MHA kernel". A Hopper-oriented kernel-test
  change silently captured the Blackwell perf-sanity run.
- **Fix mechanism:** drop the whole pattern block (the diff is 0
  additions / 23 deletions in `pytorch_model_config.py`) so Blackwell
  resolves AUTO → `TRTLLM` again. Note the remedy is *removing* a pin, not
  re-pinning a value: per the PR, Hopper's AUTO already resolves to
  `TRITON`, "matching the previous explicit pin", so the resolver was
  arch-correct on both and the explicit pin could only ever be redundant
  or wrong.
- **Detection signal:** for a tripped MoE perf bar, check whether an
  override pattern is broader than the test it was written for, and
  whether it contradicts the AUTO resolution for the arch under test:
  `grep -n -B3 -A20 "gpt_oss_20b_fp4-bench-pytorch-float4" tests/integration/defs/perf/pytorch_model_config.py`,
  then compare against `ModelConfig.resolve_moe_backend`. The cross-arch
  discriminator is diagnostic on its own: a pin that agrees with AUTO on
  the filing arch is a **no-op there and a regression elsewhere**, so the
  same test id regresses on one arch and not the other — the H200
  companion filing 6144334 failed to reproduce across 12 runs on 6 nodes
  (per that NVBug) while the B200 filing reproduced large.
- **Prevention/guard:** none added by the fix. These `patterns` entries are
  matched by model label with no GPU-arch scoping, so the next
  arch-specific workaround can capture another arch's bar just as
  silently. Gap: scope config-override blocks to the arch they were
  written for, or assert at test setup that an explicit `moe_config.backend`
  does not contradict what the AUTO resolver would pick for the arch under
  test.
- **Generalizes to:** `pattern-measurement-not-product` — the bar tripped
  because of the *test config*, not the product. Carries to: an
  over-broad test-config pattern capturing a workload it was never
  written for; a per-arch workaround applied globally, where the arch it
  was written for shows nothing; any explicit pin that freezes a knob an
  AUTO/heuristic resolver already gets right per arch; and the sibling
  case [perf-test-wrong-moe-backend](perf-test-wrong-moe-backend.md) —
  the same family of defect in the opposite direction, where the fix was
  to *add* the recommended backend pin rather than remove a wrong one.
