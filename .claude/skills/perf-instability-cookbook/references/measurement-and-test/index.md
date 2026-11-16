# Instability Cookbook — Measurement & test

This module is the **measurement apparatus itself**, not the product: the
perf-sanity harness and benchmark client under `tests/integration/defs/perf/`
(`test_perf_sanity.py`, the `parse_gen_worker_device_step_time` log parser, the
perf-DB `to_db_data()` / `to_match_keys()` bookkeeping) plus the test-side
determinism primitives it drives —
`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` and the synthetic-acceptance-rate
support in `tensorrt_llm/_torch/speculative/interface.py`. The compiled model in
every case here is stable; the number CI reports is not, either because the
metric summarises an RNG-coupled quantity or because the parser read a log the
writer had not finished flushing. These bugs are usually invisible in
single-node interactive runs and only appear under a scheduler, a cross-node
filesystem, or a suite-wide harness flag.

First thing to check: A/B the *measurement path* while holding the workload
fixed — one rep vs `--run-count`, single-node vs disagg, flag on vs off — and
compare the parsed iteration count against the run's expected iteration count
rather than comparing means.

## Recurring patterns in this module

- **Metric with hidden RNG** — the metric summarises a codepath that is not a
  pure function of its input. Spec-decode acceptance length is the canonical
  case: mean-of-N reads a different distribution every rep, so the regression
  gate flaps with no code change. The fix is a deterministic ground truth
  injected test-side (`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS`, set per
  `server_config` in the yaml, plus fractional synthetic ARs so the forced mode
  can sit at the real operating point) with the CI baseline pinned to it.
  _(Instances: [force accepted-token count in spec-decode perf test](force-num-accepted-tokens-in-spec-perf-test.md),
  [fractional synthetic acceptance rates](fractional-synthetic-acceptance-rates.md),
  [`--ignore-eos` with spec decoding](ignore-eos-with-spec-decoding.md).)_
  The third instance is the inverse warning and the one to read first, because it
  needs no new knob to reproduce: a harness flag applied to *pin* one metric can
  randomize another. `--ignore-eos` is what makes plain-decode throughput
  comparable across reps by forcing every request to `osl` tokens, and it is
  exactly what destabilizes acceptance length, because past the natural EOS the
  draft model's agreement with the target is arbitrary. Before applying a
  suite-wide "make it deterministic" flag, ask which quantity the case actually
  measures — and record the flag in the perf DB's match keys so runs measured
  with and without it can never be compared.
- **Cross-node log-flush race** — the parser reads a log written on a different
  node and returns before the last decode iterations have drained. Fixed
  **twice on the same log**, and the pair is the lesson: round 1 polled until the
  usable-line count was stable across two reads, which a writer-owned
  `srun` aggregate fd can satisfy mid-flush — so the parser stopped raising and
  started silently averaging a partial run of iterations, turning a loud failure
  into a quiet bias. Round 2 replaced the heuristic with the writer's own
  end-of-write sentinel (a `gen_server_{i}.done` touched after the `srun` is
  reaped). Read them in order; the transferable rule is that a completeness
  check must come from the writer, not be inferred by the reader — and when you
  must infer, count rows, because a mean over a prefix looks plausible.
  _(Instances: [gen_only prev_device_step_time race](gen-only-prev-device-step-time-race.md) — round 1,
  [gen_only log-flush sentinel](gen-only-log-flush-sentinel.md) — round 2.)_

Retired from this module on 2026-08-12: **process-lifetime cache binds env** — an
`os.environ` read inside a `@cache`-decorated body resolves an env-controlled
configuration once per process, so later tests in the same pytest process cannot
flip it. The trap is real, but its only instance (PR #10730; nvbugs 5680911,
5698292, 5710045, 5758449) was a CI test-isolation fix whose observable was
unit tests silently no-op'ing, with no perf metric involved. Keep the
mechanism in mind when a knob "has no effect" in a directory-level sweep; do not
expect a case here.

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [Force accepted-token count in spec-decode perf test](force-num-accepted-tokens-in-spec-perf-test.md) | spec-decode (`mtp>0`) throughput moves rep-to-rep and release-to-release with no code change because the accepted-token count per iter is a per-run draw; the PR states no magnitude | metric-with-hidden-rng |
| [Fractional synthetic acceptance rates](fractional-synthetic-acceptance-rates.md) | random-input spec-decode benchmarks report meaninglessly-low acceptance rates that move rep-to-rep on the same seed; integer-only forced ARs cannot reach the real operating point | metric-with-hidden-rng |
| [`--ignore-eos` forced generation past EOS in spec-decode cases](ignore-eos-with-spec-decoding.md) | acceptance rates unstable rep-to-rep at fixed config while throughput at fixed acceptance length is stable — no percentage, model or platform is stated anywhere, do not attach one | metric-with-hidden-rng |
| [gen_only disagg parser races NFS flush of the gen worker log](gen-only-prev-device-step-time-race.md) | intermittent `RuntimeError: … is missing 'prev_device_step_time' in gen_server_*.log`; the metric is never appended and the gate raises on a good run (round 1) | cross-node-log-flush-race |
| [gen_only log-flush sentinel](gen-only-log-flush-sentinel.md) | `mean_gen_worker_per_iter_device_step_time` silently averaged a truncated log with **no error raised**; 20 `disagg_upload-gen_only-*` cases waived on this mode alone (round 2, supersedes the row above) | cross-node-log-flush-race |
