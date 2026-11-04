# Regression Cookbook — Measurement & test

This module is the harness, not the product: the perf-sanity / QA test
definitions and their config patterns, `trtllm-bench` and its warmup and sampler
options, the launcher and process bring-up around a measured run, and the
container/Jenkins environment a probe executes in. A defect here trips a perf bar
with the product unchanged — the test selected a different backend, applied a knob
the case never asked for, measured a partially compiled model, or never got the
server to ready. **Check this module FIRST when triaging a perf-CI bar failure**:
it is the cheapest hypothesis to eliminate, and every case here was originally
investigated as a product regression.

## Recurring patterns in this module

- **Measurement, not product** — before bisecting, confirm the test still selects
  the recommended backend/config for the hardware and that the measured regime is
  stable (not host-bound low concurrency, not including warmup/capture cost in
  measured iterations). Three sub-shapes recur in this module and all five
  instances below are one of them. *(i) The harness applied a knob the case never
  asked for*: a helper that returns config for **every** test rather than for the
  tests that requested it changes what is measured suite-wide, and the affected
  cases have no way to tell — the value is not in their own config. Read the
  *emitted command line* out of the log, not the test id. *(ii) An over-broad
  selection pattern pinned the wrong backend*: the test, not the product,
  regressed. *(iii) The environment produced the number*: a container mount, a
  shared cache directory or a rank-visible `$HOME` can make a case run slowly with
  no commit in range, and the gap then fails to reproduce on a clean re-measure —
  when a reported regression will not reproduce at ≥3 reps, stop bisecting and
  diff the launch environment; the fix often lands in Jenkins/groovy, not in the
  product.
  _(Instances: the wrong MoE backend; the GPT-OSS 20B backend pin; sampler options
  applied to every case; multi-GPU bench under-warmup; the container-mounted
  `$HOME` that poisoned the Triton cache. A sixth — a bar that measured a
  host-bound low-concurrency regime (nvbug 6192201) — was removed on
  2026-08-12; "not host-bound low concurrency" survives in the advice above
  precisely because it cost a real investigation.)_
- **The server never became ready** — a launcher / IPC / process-bring-up change
  can trip a perf bar with **no compute commit in range**, which is the blind spot
  of the bullet above: the product *did* change, just not anywhere a profile would
  look. A truncated or never-started run reaches the QA comparison as a percentage
  regression rather than as an error, so ask "did the server reach ready?" before
  "which kernel got slower?".
  _(Instance: the IPC HMAC key passed by fd — landed and reverted **twice**, and it
  polluted three unrelated investigations in these cookbooks before being
  recorded here.)_
- **Warmup coverage gap** — the harness can create one on its own: a `--warmup`
  count that does not scale with rank count leaves compile cost inside the
  measured window, so a multi-rank case measures a partially compiled model.
  _(Instance: multi-GPU bench under-warmup.)_

_Cross-node variance belongs on this checklist too, but its case lives in the
`communication/` module (`allreduce-host-overhead-small-model-tp.md`). On a
shared multi-node cluster, pin every probe in the window to ONE node with ≥3 reps
before bisecting._

## Cases

| Case | Symptom (signal) | Class |
|------|------------------|-------|
| [IPC HMAC key passed by file descriptor deadlocks the benchmark launcher](ipc-hmac-key-via-fd-breaks-bench.md) | round 2 hangs 30–90 min after `start MpiSession`, all workers at 0% GPU; the bar reports a % drop | measurement-artifact |
| [trtllm-bench warmed up fewer times than there are ranks](multi-gpu-bench-warmup-too-few.md) | multi-GPU perf-sanity cases read low and unstable; the PR description is empty | measurement-artifact, warmup-jit-gap |
| [A container-mounted `$HOME` poisoned the Triton cache; the gap never reproduced](nixl-ctx-only-gap-not-reproduced-home-mount.md) | a NIXL disagg `ctx_only` throughput gap; the fix is a CI job-configuration change, not a product change | measurement-artifact |
| [perf-sanity applied top_k/top_p/temperature to every case](perf-sanity-sampler-options-applied-to-all-cases.md) | broad simultaneous drop across cases sharing only the harness; no per-case percentage stated | measurement-artifact |
| [An over-broad perf-test pattern pinned GPT-OSS 20B to the TRITON MoE backend](perf-test-gpt-oss-20b-moe-backend-pin.md) | `gpt_oss_20b_fp4-bench-pytorch-float4` regressed on Blackwell; product unchanged | measurement-artifact, kernel-selection-regression |
| [GPT-OSS 120B perf test ran the non-recommended CUTLASS MoE backend](perf-test-wrong-moe-backend.md) | GPT-OSS 120B max-throughput perf tests below best; the measurement regressed, not the product | measurement-artifact |
