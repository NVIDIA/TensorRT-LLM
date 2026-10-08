---
id: case-gen-only-log-flush-sentinel
type: instability-case
family: measurement-determinism
module: measurement-and-test
maturity: full
instability_class: [cross-node-log-flush-race]
signals: [perf-ci-bar-flake, rep-to-rep-variance, test-parser-missing-line]
subsystems: [perf-test-harness, disagg-serve]
introduced_via: [prior-fix-side-effect]
phase: [decode]
patterns: [pattern-cross-node-log-flush-race]
nvbugs: []
commits: ["99bdffc4c39b"]
success_prs: [16717]
failed_prs: []
---

# `gen_only` metric averaged a truncated log: settle-poll heuristic replaced by an end-of-write sentinel

> Part of the [Measurement & test instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `99bdffc4c39b` · PR #16717 — Wait for gen-log
  end-of-write sentinel before parsing per-iter step time. **Round 2 of
  [gen_only prev_device_step_time race](gen-only-prev-device-step-time-race.md)**
  (`e9402ab59dab` · PR #15108): same log, same race, and the round-1 mitigation
  is what this case supersedes. Read that case first — this one is only
  intelligible as the correction to its heuristic. The PR title names nvbugs
  6487040 / 6487036, but **both are functional bugs** (per the bug records,
  checked 2026-08-12: parser failures on `prev_device_step_time` in the disagg
  gen log — one a mismatched parse, one the field missing), so they are named
  in prose only and `nvbugs:` stays empty per this cookbook's admission rule.
- **Symptom (variance signature):** the disagg `gen_only` perf-sanity metric
  `mean_gen_worker_per_iter_device_step_time` was intermittently computed from
  a **truncated** `gen_server_{i}.log` — so the reported number was a mean over
  a *partial* run of iterations, varying rep to rep with how much of the log
  had drained, with **no error raised**. Surfaced as perf-sanity CI flake on
  `disagg_upload-gen_only-*` cases (20 of them waived on this failure mode
  alone).
- **Root cause:** the round-1 fix polled until the usable-line count was
  "unchanged across two consecutive reads" and treated that as flush-complete.
  That heuristic is unsound when the writer is an `srun`-owned **aggregate fd**:
  the gen `srun` `&>`-redirects every TP rank into one file, so a pause in
  arriving lines is not evidence of EOF, and the poll "could latch onto a
  mid-flush prefix — silently averaging a partial run of iterations" (PR body).
  Round 1 converted a loud failure (parser returns `None` → gate raises) into a
  **quiet bias**, which is strictly worse: the same race now moved the number
  instead of dropping it.
- **How introduced:** `prior-fix-side-effect` — PR #15108's settle-poll. Note
  the *metric definition* it parses had also just changed: PR #16298 restricted
  the mean to steady-state iterations, and its unbucketable-
  `num_generation_tokens` path is what produced the `None` metric that #16717
  keeps an all-iter Welford fallback for. #16298 is the change under repair
  here, not a second cause.
- **Fix mechanism:** replace the heuristic with a real **end-of-write
  sentinel**. Each gen `srun` runs in the foreground of a backgrounded subshell
  that `touch`es `gen_server_{i}.done` immediately after the `srun` returns;
  because the `srun` owns the aggregate fd, the sentinel fires strictly *after*
  reap, i.e. after the log is fully flushed. The BENCHMARK `srun` defers its
  parse out of the client loop, writes `benchmark_status` in `finally` (which
  releases the gen workers so their `srun` can exit — this handshake is why the
  wait is not circular), blocks in `wait_for_gen_log_sentinels()` bounded by
  `self.timeout`, then parses each client log exactly once. A stale sentinel
  from a reused output dir is removed first. The settle loop and its
  `settle_timeout` / `poll_interval` kwargs are deleted.
- **Detection signal:** a perf metric that is *plausible but low* with no
  correlated code change, where the same config reports different means across
  reps; compare the parsed iteration count against the run's expected iteration
  count — a mean over a prefix is only detectable by counting rows, never by
  looking at the value. `grep -n 'wait_for_gen_log_sentinels\|gen_server_.*\.done'
  tests/integration/defs/perf/test_perf_sanity.py jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh`
  to confirm the sentinel path is in place, and `grep -c 'settle_timeout'` to
  confirm the heuristic is gone.
- **Prevention/guard:** the 20 un-waived `disagg_upload-gen_only-*` cases are
  the guard — the fix is re-validated end-to-end in pre-merge CI. Scope was
  deliberately limited to the path the root cause explains: aggregated
  (`aggr_upload-*`), `disagg_upload-e2e` and `disagg_upload-ctx_only` waivers
  were left untouched, since aggregated serving produces no `gen_server_{i}.log`
  and so cannot hit this race. The transferable rule: **a completeness check on
  a file must come from the writer, not be inferred by the reader.** Any
  reader-side "it stopped changing, so it must be done" test is a heuristic; if
  the writer can signal (sentinel file, exit code, reaped pid), require that
  signal. And when replacing a loud failure with a tolerant one, verify the
  tolerant path cannot silently produce a *wrong number* — that trade is the
  actual defect here.
- **Generalizes to:** `pattern-cross-node-log-flush-race`; carries to every
  harness that parses a log written by another process or node (throughput
  logs, KV-hit-rate logs, spec-decode acceptance logs), to any aggregate fd
  shared by N ranks where per-rank quiescence ≠ completeness, and — most
  broadly — to every "poll until stable" idiom standing in for an absent
  end-of-stream signal.
