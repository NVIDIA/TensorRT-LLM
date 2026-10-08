---
id: case-piecewise-cudagraph-far-ceiling-append
type: regression-case
family: execution-and-graph
module: cuda-graph-and-compile
maturity: full
regression_class: [cuda-graph-regression]
signals: [throughput-drop, ttft-increase, itl-increase, slower-kernel-in-trace]
subsystems: [cuda-graph]
introduced_via: [prior-fix-side-effect]
phase: [prefill]
patterns: [pattern-padding-bucket-overshoot]
nvbugs: ["6404567"]
commits: ["459839cc271f"]
success_prs: [16256]
failed_prs: []
---

# Force-appended piecewise-graph capture ceiling makes padding run the 65536-token graph

> Part of the [CUDA graph & compile regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6404567` · commit `459839cc271f` · PR #16256 — "Clamp
  piecewise cudagraph captures to the reachable ceiling instead of
  force-appending it".
- **Symptom:** `gpt-oss-120b` non-disaggregated IFB serving on GB300-NVL72
  (72× TP1 `trtllm-serve` replicas, aarch64) lost ~12% throughput, with TTFT
  more than doubling and TPOT up ~7%, moving only from the `1.3.0rc14`
  container to `1.3.0rc15` — no change to model, config, hardware or client.
  The shift is fully introduced at the rc14→rc15 step and persists unchanged
  through rc16/rc17/rc20 and main. Surfaced as a customer report,
  not a perf CI bar. PR #16256's GB300 A/B (4× TP1 servers, 896 concurrent
  streams/server, 24000 requests per arm, steady-window aggregates): rc14
  51.40 QPS / 55.4 ms TPOT vs rc15–rc20/main stock 44.3–44.7 QPS /
  64–65 ms TPOT.
- **Root cause:** the piecewise CUDA-graph capture filter *force-appended* the
  engine's reachable `num_tokens` ceiling
  (`max_batch_size * (max_seq_len - 1 - num_extra_decoding_steps)`) to the
  user's `torch_compile_config.capture_num_tokens` list. Runtime padding
  rounds each context-bearing iteration up to the nearest captured size, so
  with the bug's config — a user list topping out at 13914 and a ceiling of
  65536 — every iteration in the (13914, 65536] token gap executed the full
  65536-token graph. Per the PR, measured per-iteration device time is flat
  ~325–400 ms against 116–408 ms for true-size eager (up to 2.8×), i.e. the
  appended entry is worse than having no graph for that range at all, which
  is exactly what rc14 did. Those iterations then serialize the in-flight
  decode streams behind them, which is why a prefill-side capture bug moves
  TPOT as well as TTFT.
- **How introduced:** PR #13574 / commit `9c1869b3c0ab` — "Broader capture of
  piecewise cudagraph", the fix for nvbug 5615248 (see
  [piecewise-graph capture coverage](piecewise-cudagraph-capture-coverage.md)).
  That fix did two things: (a) drop capture candidates above the reachable
  ceiling, and (b) force-append the ceiling itself so token counts in the gap
  below it would get *a* graph. Half (a) was correct; half (b) is this
  regression. Named as the culprit in the fix PR's description and in the
  NVBug.
- **Fix mechanism:** replace drop-and-append with **clamping** in
  `_filter_piecewise_capture_num_tokens`
  (`tensorrt_llm/_torch/pyexecutor/model_engine.py`): candidates above the
  reachable ceiling are lowered *to* the ceiling (a requested 128 becomes 127
  when only 127 is recordable) and no capture size beyond the user's list is
  ever invented. Configs with no oversized entries keep exactly their own
  list, i.e. pre-`1.3.0rc15` behavior. The nvbug 5615248 config's capture list
  is preserved bit-for-bit, so the original fix's coverage is not lost. With
  the fix, rc15/rc18/rc20 return to 51.6–52.3 QPS / ~55 ms TPOT, and a
  same-binary main A/B (identical native library SHAs, Python delta only)
  reads 52.15 vs 44.63 QPS and 54.4 vs 64.5 ms TPOT.
- **Detection signal:** the effective capture list contains an entry far above
  its predecessor, and per-iteration device time is *flat* at the largest
  captured shape instead of tracking true token count. Check the ratio between
  adjacent entries in the list the engine actually keeps rather than the list
  in the YAML —
  `python -c "from tensorrt_llm._torch.pyexecutor.model_engine import _filter_piecewise_capture_num_tokens as f; print(f(<capture_num_tokens>, max_num_tokens=<n>, max_batch_size=<b>, max_seq_len=<s>))"`
  — and on affected builds
  `grep "exceeds reachable ceiling" <serve log>`, whose pre-fix wording
  ("Capturing the ceiling itself") names the injected entry outright. A/B
  confirmation both ways is cheap and was done on this bug: adding 65536 to
  rc14's list reproduces the full regression, and neutralizing the appended
  65536 on rc15/rc20 restores rc14-level QPS/TTFT/TPOT.
- **Prevention/guard:** PR #16256 added two unit tests to
  `tests/unittest/llmapi/test_llm_args.py::TestPiecewiseCudaGraphCaptureDefaults`
  — one pinning the exact 6404567 shape (a far ceiling is never invented) and
  one that multiple oversized candidates collapse to a single clamped ceiling
  — updated 3 existing tests to clamp semantics, and kept the nvbug 5615248
  regression tests passing unchanged. General guard: a coverage fix that
  *adds* a bucket to a pad-to-nearest scheme must be measured on the gap it
  creates, not only on the shape it was meant to cover; and an explicit
  user-supplied list is a perf contract the engine should clamp into, never
  extend.
- **Generalizes to:** `pattern-padding-bucket-overshoot` — in any
  pad-up-to-nearest-bucket scheme, one bucket far above its neighbour turns
  the whole gap below it into worst-case work, so adding a bucket is not
  monotonically safe. Carries to: decode CUDA-graph `batch_sizes` lists where
  an appended `max_batch_size` sits far above the previous entry (padding-
  enabled capture pads every batch in between); torch.compile / autotuner
  shape buckets whose top bucket is far from the second; attention chunk or
  KV-page granularity chosen "generously" so short sequences pay the large
  shape; and any fix that closes a *coverage* gap by widening a bucket set
  instead of tightening it.
