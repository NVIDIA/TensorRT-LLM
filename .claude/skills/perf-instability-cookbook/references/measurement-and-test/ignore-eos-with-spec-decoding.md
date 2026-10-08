---
id: case-ignore-eos-with-spec-decoding
type: instability-case
family: measurement-determinism
module: measurement-and-test
maturity: full
instability_class: [metric-with-hidden-rng]
signals: [rep-to-rep-variance, acceptance-length-drift, perf-ci-bar-flake]
subsystems: [spec-decode, perf-test-harness]
introduced_via: [preexisting]
phase: [decode]
patterns: [pattern-metric-with-hidden-rng]
nvbugs: ["6143945"]
commits: ["ac0be4774804"]
success_prs: [14347]
failed_prs: []
---

# `--ignore-eos` forced generation past EOS, making spec-decode acceptance rates unstable

> Part of the [Measurement & test instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6143945` · commit `ac0be4774804` · PR #14347 — gates
  `--ignore-eos` on whether spec decoding is enabled. **The PR names no bug**; the
  linkage comes from the bug.
- **Symptom:** rep-to-rep variance in a spec-decode perf-sanity case, expressed
  through acceptance length rather than through latency. The only symptom statement
  anywhere is the comment the PR adds to the harness: "`--ignore-eos` must be off
  when spec decoding is enabled: forcing generation past EOS produces unstable
  acceptance rates." **No percentage, model or platform is stated** in the PR — do
  not attach one.
- **Root cause:** the benchmark client passes `--ignore-eos` unconditionally so
  that every request generates exactly `osl` tokens, which is what makes throughput
  comparable across reps. With spec decoding on, that stability guarantee inverts:
  past the natural EOS the model is generating text it has no distribution for, so
  the **draft model's agreement with the target becomes arbitrary** and acceptance
  length — the quantity the case is measuring — is driven by whatever the model
  wanders into. The knob that removes variance from a plain-decode measurement is
  the knob that injects it into a spec-decode one, which is why it survived: it is
  correct, and load-bearing, everywhere else in the suite.
- **How introduced:** `preexisting` — the flag was applied suite-wide before spec
  decode cases existed, and no gate was added when they were.
- **Fix mechanism:** thread the fact through the harness rather than special-casing
  a test id. `ClientConfig` gains `spec_decoding: bool = False`; both command
  builders — `_to_sa_benchmark_cmd` and `_to_default_benchmark_cmd` — emit
  `--ignore-eos` only when it is false; and `"b_eos"` is recorded in
  `to_db_data()` **and** `to_match_keys()`. That last part is the detail to copy:
  putting the flag in the match keys means results measured with and without
  `--ignore-eos` cannot be silently compared against each other in the perf
  database, so the fix cannot corrupt the historical baseline it changes.
  Single file: `tests/integration/defs/perf/test_perf_sanity.py` (+23/−2).
- **Detection signal:** `grep -n -- "--ignore-eos" <bench_cmd_or_log>` on a
  spec-decode case — present ⇒ pre-fix behaviour. Statically:
  `git grep -n "ignore_eos\|ignore-eos" tests/integration/defs/perf/` and check
  every emitter is gated. The variance signature is acceptance length differing
  rep-to-rep at fixed config while *throughput at fixed acceptance length* is
  stable; if the run also generates well past a natural stopping point, this is it.
- **Prevention/guard:** **no test was added.** The durable guard is the `to_match_keys()`
  entry, which prevents cross-comparison rather than recurrence. Rule to carry: a
  harness flag that pins one metric may randomize another — before applying a
  suite-wide "make it deterministic" knob, ask which quantity the case actually
  measures. Forcing generation past EOS is never valid when the metric depends on
  model agreement.
- **Generalizes to:** `pattern-metric-with-hidden-rng`; joins
  `case-force-num-accepted-tokens-in-spec-perf-test` (pin acceptance explicitly)
  and `case-fractional-synthetic-acceptance-rates` (a synthetic rate that cannot be
  represented) as the third face of the same problem: **spec-decode acceptance
  length is an RNG-coupled metric, and every perf case that reports it needs its
  randomness sources enumerated**, including the ones the harness adds.
