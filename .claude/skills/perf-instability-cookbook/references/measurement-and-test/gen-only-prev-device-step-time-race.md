---
id: case-gen-only-prev-device-step-time-race
type: instability-case
family: measurement-determinism
module: measurement-and-test
maturity: full
instability_class: [cross-node-log-flush-race]
signals: [perf-ci-bar-flake, test-parser-missing-line]
subsystems: [perf-test-harness, disagg-serve]
introduced_via: [incomplete-coverage]
phase: [decode]
patterns: [pattern-cross-node-log-flush-race]
nvbugs: []
commits: ["e9402ab59dab"]
success_prs: [15108]
failed_prs: []
---

# `gen_only` disagg perf-sanity parser races NFS flush of gen worker log

> Part of the [Measurement & test instability cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** commit `e9402ab59dab` · PR #15108 — Fix gen_only
  missing `prev_device_step_time` race in perf sanity.
  **Round 1 of two.** This fix was later found insufficient and was replaced
  by a writer-side sentinel: read
  [gen_only log-flush sentinel](gen-only-log-flush-sentinel.md) (PR #16717)
  after this case. Do not copy the poll-until-stable mechanism below without
  reading why it fails.
- **Symptom (variance signature):** `gen_only` disaggregated
  perf-sanity tests intermittently hard-failed with
  `RuntimeError: gen_only test Server N Client M is missing 'prev_device_step_time' in gen_server_*.log`.
  Same test, same input, same nodes, same seed — pass / fail was
  purely a function of how fast NFS drained the last few iters of the
  gen worker's log. Downstream, the `d_mean_gen_worker_per_iter_device_step_time`
  metric was never appended and the gen_only regression gate raised.
- **Root cause:** a **read-after-write race**, not a benchmark problem.
  `parse_gen_worker_device_step_time` reads `gen_server_*.log` **once**
  immediately after the benchmark client returns. But the gen worker
  writes that log on a *different node*, and is deliberately kept
  alive (it blocks on the `benchmark_status` file). When the client
  returns, the decode iterations are done but their log lines are
  often still flushing across NFS. The `[start_offset, EOF]` slice
  the parser saw contained **zero `iter >= 5` lines** → the parser
  returned `None` → the metric summary line was never appended → the
  gen_only gate (this test's only regression signal) raised on a
  perfectly-good run.
- **How introduced:** the parser was authored for single-node
  benchmarks where writer-flush and reader-open are locally
  synchronous; disagg gen_only put writer and reader on different
  nodes and inherited the single-node parse timing.
- **Fix mechanism:** poll the slice until the usable-line count is
  non-zero **and stable across two consecutive reads** (i.e. the
  cross-node flush has drained), bounded by `settle_timeout`
  (default 90 s). This waits for the data to become **visible**
  rather than for the file/process to "finish" (it can't — the worker
  outlives the parse by design). Secondary latent crash also fixed:
  open the log with `errors="replace"` so `tqdm` model-load progress
  bars (partial multibyte glyphs under interleaved writes) don't
  raise `UnicodeDecodeError` mid-scan. No change to metric semantics:
  steady-state mean over `iter >= 5` is identical once the lines are
  present.
- **Detection signal:** perf-sanity CI reports
  `missing 'prev_device_step_time'` on a fraction of `gen_only` runs
  with no correlated code change; the gen worker log on the cluster
  DOES contain the expected iter lines when inspected minutes later;
  `grep -n 'settle_timeout\|prev_device_step_time' tests/integration/defs/perf/`
  to confirm the poll-until-stable path is in place.
- **Prevention/guard:** any perf-harness parser that reads a log
  produced on a different node must not **read once**. The "process is
  still alive → data is still coming" invariant does not hold for
  cross-node NFS. But note the correction round 2 makes to the guard
  stated here originally: "poll until stable" is **not** a sufficient
  substitute, because the writer is an `srun`-owned *aggregate* fd, so a
  pause in arriving lines is not evidence of EOF. Polling merely turned a
  loud failure (parser returns `None`, gate raises) into a quiet bias —
  a mean over a truncated prefix, which looks plausible. The correct
  guard is the writer's own end-of-write signal; see
  [gen_only log-flush sentinel](gen-only-log-flush-sentinel.md).
- **Generalizes to:** `pattern-cross-node-log-flush-race`; carries to
  every disagg / multi-node metric parse (throughput logs, KV-cache
  hit-rate logs, prev_device_step_time for spec-decode, worker
  lifecycle logs), and to every write-once-elsewhere / read-once-here
  measurement plane the harness inherits from a single-node design.
