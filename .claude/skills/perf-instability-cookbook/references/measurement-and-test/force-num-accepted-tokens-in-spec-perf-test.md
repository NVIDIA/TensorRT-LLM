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

- **Provenance:** nvbugs `6162561` / `6248724` · commit `26c099f52dda` ·
  PR #14438 — Add `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` in
  spec-decoding perf test. Note the PR itself names **no** bug — its title is
  tagged `[None][test]` — so a PR→bug lookup finds nothing.
  Related to
  [case-fractional-synthetic-acceptance-rates](fractional-synthetic-acceptance-rates.md)
  (#13569) which supplies the fractional-AR primitive this test relies
  on.
- **Symptom (variance signature):** spec-decode perf-sanity throughput moved
  between reps and releases with no code change, because the number of
  accepted draft tokens per iteration was itself a per-run draw: each rep drew
  a different distribution of accepted tokens, so mean throughput moved and
  the regression gate flapped. The PR describes the fix ("stabilize
  accepted-token count") rather than the symptom and **states no magnitude,
  model or platform** — do not attach one. The tell to look for is that the
  movement is confined to cases with spec decoding on (`mtp>0`) and its sign
  and size change from one comparison to the next, a spread no code change
  explains.
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
  turning the env var on across the mtp perf-sanity matrix exposed a CUDA
  illegal memory access (nvbug `6342840`): per PR #15797, the forced count
  inflated `num_accepted_tokens` during eager CUDA-graph warmup, where the
  dummy requests' KV / MTP-pool / draft-token buffers are not populated, so
  downstream C++ MTP ops indexed out of bounds. PR #15797 (`spec_metadata=None`
  kwarg on `SpecWorkerBase._apply_force_accepted_tokens`, merged 2026-07-01)
  fixed it, and PR #15827 (merged 2026-07-02) un-waived the mtp perf-sanity
  cases that had been waived meanwhile (18 lines it attributes to #15797, the
  rest for CI recheck). So the cost of pinning an RNG in the harness was a
  month-long hole in the very matrix it was meant to stabilize — budget for a
  soak on the forced path before enabling it fleet-wide.
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
