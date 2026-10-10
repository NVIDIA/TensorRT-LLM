---
id: case-force-num-accepted-tokens-in-spec-perf-test
type: instability-case
family: measurement-determinism
module: measurement-and-test
maturity: full
instability_class: [metric-with-hidden-rng]
signals: [rep-to-rep-variance, acceptance-length-drift]
subsystems: [spec-decode, perf-test-harness]
introduced_via: [incomplete-coverage]
phase: [decode]
patterns: [pattern-metric-with-hidden-rng]
nvbugs: ["6162561", "6248724"]
commits: ["26c099f52dda"]
success_prs: [14438]
failed_prs: []
---

# Spec-decode perf test needs `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` to pin acceptance length

> Part of the [Measurement & test instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6162561` (and the sibling filing `6248724` —
  **not** a duplicate: neither bug carries a duplicate marker, they are two
  independent QA release-regression filings, three weeks and one GPU
  generation apart, answered by the same PR) ·
  commit `26c099f52dda` · PR #14438 — Add
  `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` in spec-decoding perf
  test. Note the PR itself names **no** bug — its title is tagged
  `[None][test]` — so a PR→bug lookup finds nothing; both bugs name
  #14438 as their fix, which is the only link.
  Related to
  [case-fractional-synthetic-acceptance-rates](fractional-synthetic-acceptance-rates.md)
  (#13569) which supplies the fractional-AR primitive this test relies
  on.
- **Symptom (variance signature):** both bugs were filed as
  release-to-release **regressions** in the QA Inference Time metric, and
  both were answered as acceptance-length volatility rather than a mean
  shift — for nvbug `6248724`, an env var to keep acceptance length from
  fluctuating is the entire diagnosis. On nvbug `6162561` (GB200-<cluster>,
  1.3.0rc14 → rc15) the reported rows were +221.21 %
  (`disagg-e2e-gb200_deepseek-v32-fp4_32k4k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL`,
  19234.641 → 61784.270), +51.94 % (the `kimi-k25-thinking-fp4_8k1k_con4`
  mtp3 case) and +12.57 % (the `deepseek-r1-fp4_1k1k` mtp3 case) against
  a 5 % regression bar; nvbug `6248724` reported that *same*
  kimi-k25-thinking mtp3 case at +6.85 % on GB300-<cluster> (4006.460 →
  4280.710, 1.3.0rc16 → rc17). Two tells that this is variance and not a
  regression: every attributed row carries `mtp3` (spec decoding on) —
  the one `mtp0` row in `6162561`'s table was explicitly *not* attributed
  to this fix — and one case's "regression" reads 51.94 % in one release
  pair and 6.85 % in another, a spread no code change explains. Each rep
  drew a different distribution of accepted draft tokens per iter, so the
  mean throughput moved with no code change and the regression gate
  flapped.
- **Root cause:** the perf test measured spec-decoding throughput
  end-to-end but did not control the acceptance length — the very
  quantity that turns "how many draft tokens per iter" into "how many
  useful tokens per iter". With `--ignore-eos` and random-ish inputs,
  the accepted count is a random variable whose mean-of-N over a
  short benchmark is a hidden RNG in the metric.
- **How introduced:** spec-decoding perf-sanity was authored around
  the model's actual acceptance behaviour on the benchmark prompts —
  fine as a smoke test, but the noise term never got the same
  treatment as latency / memory noise (fixed seeds, `--ignore-eos`).
- **Fix mechanism:** (1) always pass `--ignore-eos` for spec-decode
  perf tests. (2) stabilize the accepted-token count via
  `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`, set per `server_config`
  in the yaml (never computed in code, so a config-time change is a
  visible diff not a runtime surprise). (3) new
  `d_al` (acceptance-length) metric, with `l_force_num_accepted_tokens`
  added as a baseline match key so different forced values match
  separately. (4) new `d_mean_gen_worker_per_iter_device_step_time`
  metric for gen_only tests — gen_only regression gates on this
  instead of throughput, removing yet another source of hidden RNG.
  (5) yaml schema cleanup — agg yamls move env to per-server-config
  `server_env_var`; disagg adds spec-decode env to `worker_env_var`
  only.
  **The stabilizer itself then broke, and that is part of the record:**
  turning the env var on across the mtp perf-sanity matrix exposed
  nvbug `6342840` (a functional bug, so deliberately absent from
  `nvbugs:` above) — `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`
  triggered a CUDA illegal memory access in MTP-enabled
  disaggregated/aggregated perf-sanity on GB200 + GB300. It took
  PR #15797 (`spec_metadata=None` kwarg on
  `SpecWorkerBase._apply_force_accepted_tokens`, merged 2026-07-01) to
  fix, and 37 mtp cases sat waived until PR #15827 un-waived them
  (18 fixed by #15797 + 19 for CI recheck, merged 2026-07-02). So the
  cost of pinning an RNG in the harness was a five-week hole in the
  very matrix it was meant to stabilize — budget for a soak on the
  forced path before enabling it fleet-wide.
- **Detection signal:** perf-sanity CI for spec-decode reporting rep-
  to-rep throughput variance uncorrelated with any code change; the
  `d_al` metric moving by more than a few % across reps of the same
  test with identical inputs;
  `grep -nE 'TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS|l_force_num_accepted_tokens' tests/integration/defs/perf/`
  to confirm the forced-AR path is used.
- **Prevention/guard:** any perf test whose metric depends on a
  discrete-event process (acceptance, cache hit rate, batch fill
  rate, page eviction) must **pin the RNG** at the benchmark harness
  level, not rely on averaging over N to converge. Baseline match keys
  must include every forced-configuration knob so different forced
  values don't collapse into one baseline row.
- **Generalizes to:** `pattern-metric-with-hidden-rng`; carries to
  every perf metric summarising over a stochastic per-iter behaviour
  (KV-cache hit rate perf tests, MoE routing balance perf tests,
  batched prefill sharing tests). Also — this is why a perf-instability
  commit search must include test-side knobs: subject-line grep for
  stabil/instab misses `[test]`/`[feat]`-tagged determinism-forcing PRs
  like this one.
