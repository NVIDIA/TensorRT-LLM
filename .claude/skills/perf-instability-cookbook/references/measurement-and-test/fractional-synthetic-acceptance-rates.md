---
id: case-fractional-synthetic-acceptance-rates
type: instability-case
family: measurement-determinism
module: measurement-and-test
maturity: full
instability_class: [metric-with-hidden-rng]
signals: [rep-to-rep-variance, acceptance-length-drift]
subsystems: [spec-decode, perf-test-harness]
introduced_via: [preexisting]
phase: [decode]
patterns: [pattern-metric-with-hidden-rng]
nvbugs: []
commits: ["a13c19be3482"]
success_prs: [13569]
failed_prs: []
---

# Fractional synthetic acceptance rates — deterministic AR ground truth for random-input benchmarks

> Part of the [Measurement & test instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** JIRA `TRTLLM-12390` · commit `a13c19be3482` ·
  PR #13569 — Support fractional synthetic acceptance rates.
  Foundation for
  [case-force-num-accepted-tokens-in-spec-perf-test](force-num-accepted-tokens-in-spec-perf-test.md)
  (#14438), which consumes this primitive as the CI stabilizer.
- **Symptom (variance signature):** spec-decode perf benchmarks with
  random inputs report meaninglessly-low acceptance rates whose value
  moves rep-to-rep on the same seed. Any downstream metric that scales
  with acceptance (throughput, tokens/s, effective latency) inherits
  the variance and produces false regression signals. Prior to this
  PR only integer synthetic ARs were supported, which was too coarse
  to sweep the operating region where the model's actual AR sits.
- **Root cause:** perf debug for spec-decode needed a way to hold
  acceptance rate constant across reps while varying other axes, but
  the synthetic-AR primitive only accepted integer values. On random
  input distributions the *real* AR is fractional, so the harness
  either had to run the model (variance) or force an integer AR that
  did not match the operating point (invalid).
- **How introduced:** the synthetic-AR feature landed as an integer
  primitive to unblock a first perf-debug use case, without a plan for
  hitting the fractional operating points production workloads sit at.
  Concretely that primitive is
  `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`, added by PR #9371
  ("Add environment variable to force spec-dec number of accepted
  tokens", merged 2025-11-26, commit `ef7ee6a94058`) and repaired for
  MTP/EAGLE by PR #9608 — a *count* of accepted tokens, so integral by
  construction. #13569 touches the same
  `tensorrt_llm/_torch/speculative/interface.py` and lands
  `tests/unittest/_torch/speculative/test_force_accepted_tokens.py`,
  which is how the fractional extension attaches to the integer
  original. Note the six-month gap: #14438 depends on #9371 for the env
  var and on this PR only for fractional targets.
- **Fix mechanism:** extend synthetic-AR generation to accept
  fractional targets, resolved by deterministic selection logic (not
  RNG at test time) — the same fractional target yields the same
  per-iter accept/reject decisions across reps. Comprehensive test
  coverage for fractional acceptance behavior.
- **Detection signal:** perf test log includes an acceptance-length
  metric that moves >5% rep-to-rep with no code change on random
  inputs; `grep -nE 'synthetic.*acceptance|fractional.*AR|forced.*acceptance' tensorrt_llm/`
  for the fractional-AR primitive. This case is the primitive; #14438
  is the CI wiring that turns it into a stable metric.
- **Prevention/guard:** every stochastic-input perf test that reports
  a metric influenced by a discrete-event process MUST have a
  deterministic ground-truth mode (forced AR / forced hit rate /
  forced eviction rate). The forced mode must sweep the region of
  interest — integer-only forced modes are insufficient for
  fractional operating points.
- **Generalizes to:** `pattern-metric-with-hidden-rng`; carries to
  every perf harness whose input distribution is synthetic and whose
  metric summarises over a stochastic per-iter process — spec-
  decoding acceptance, MoE routing balance, KV-cache reuse, batched
  prefill sharing, guided-decoding-mask hit rate.
