---
id: case-multi-gpu-bench-warmup-too-few
type: regression-case
family: measurement-and-test
module: measurement-and-test
maturity: full
regression_class: [measurement-artifact, warmup-jit-gap]
signals: [perf-ci-bar-failure, startup-time-increase, throughput-drop]
subsystems: [perf-test-config, runtime-python]
introduced_via: [incomplete-coverage]
phase: [any-phase]
patterns: [pattern-measurement-not-product, pattern-warmup-coverage-gap]
nvbugs: ["5582091"]
commits: ["4586b5f42f2c"]
success_prs: [9578]
failed_prs: []
---

# trtllm-bench warmed up fewer times than there are ranks, so multi-GPU runs measured a partially-compiled model

> Part of the [Measurement & test regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5582091` · commit `4586b5f42f2c` · PR #9578 —
  "[https://nvbugs/5582091][test] increase warmup times in testing for
  multi-gpu cases".
- **Symptom:** multi-GPU `trtllm-bench` perf-sanity cases reading low and
  unstable. **The PR description is empty** (CodeRabbit flagged exactly that, and
  there are zero human review comments) — the diff is the only public record.
- **Root cause:** the default warmup count is a small constant, independent of
  rank count. A one-time first-execution cost (e.g. a `torch.compile`) that is
  paid **per rank** needs at least as many warmup requests as ranks to cover
  every rank — which is what the fix's `num_gpus`-scaled warmup encodes. With
  fewer warmup requests than ranks, some ranks
  entered the measured window still cold, so the measurement averaged compiled and
  uncompiled ranks. The failure mode is subtle because it is not "warmup is
  missing" but "warmup ran, and covered a subset of the ranks" — a warmup loop
  counted in *requests* silently under-covers a system parallelised over *ranks*,
  since a single request need not touch every rank's first-execution path.
- **How introduced:** `incomplete-coverage`. The warmup constant was right for
  single-GPU and was never re-derived when multi-GPU cases were added.
- **Fix mechanism:** scale warmup with the rank count, in
  `get_trtllm_bench_command`:
  `if self._config.num_gpus > 1: benchmark_cmd += [f"--warmup={2 * self._config.num_gpus}"]`.
  The factor 2 reads as headroom over a `num_gpus` floor. One harness file
  (`tests/integration/defs/perf/test_perf.py`, +2 lines); nothing in the
  engine changes.
- **Detection signal:** `grep -o '\-\-warmup=[0-9]*' <bench_cmd_or_log>` and
  compare against the case's GPU count — a value below `num_gpus` on a multi-GPU
  case is this defect. Corroborate in the run log by looking for compile / JIT work
  inside the measured region rather than before it.
- **Prevention/guard:** none added — no test asserts warmup ≥ rank count. Rule to
  carry: **warmup counts must be expressed in units of whatever the cold cost is
  paid per.** If the cost is per rank, per shape, or per bucket, a
  request-counted warmup is not a guard. And an empty PR description on a
  measurement-side fix is a durable cost: the *why* is not in the public record.
- **Generalizes to:** `pattern-measurement-not-product` (the product was fine; the
  measurement was cold) and `pattern-warmup-coverage-gap` (the coverage hole is
  per-rank). Carries to every benchmark client with a fixed warmup constant, to
  TP/PP/EP scaling of any warmup grid, and to the general suspicion that a case
  which improves when you simply run it longer is under-warmed rather than slow.
