---
id: case-piecewise-cudagraph-capture-coverage
type: regression-case
family: execution-and-graph
module: cuda-graph-and-compile
maturity: full
regression_class: [cuda-graph-regression, fast-path-fallback]
signals: [ttft-increase, throughput-drop, host-time-increase]
subsystems: [cuda-graph]
introduced_via: [incomplete-coverage]
phase: [prefill]
patterns: [pattern-fast-path-silent-fallback]
nvbugs: ["5615248"]
commits: ["9c1869b3c0ab"]
success_prs: [13574]
failed_prs: []
---

# Piecewise CUDA graph capture set misses reachable num_tokens; prefill falls back to eager

> Part of the [CUDA graph & compile regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5615248` · commit `9c1869b3c0ab` · PR #13574 —
  "Broader capture of piecewise cudagraph". The PR describes itself as "part
  of a fix" for nvbug 5615248; this case covers only the piecewise-cudagraph
  capture-set commit.
- **Symptom:** With `enable_piecewise_cuda_graph=True`, prefill forward
  passes for `num_tokens` values above the largest *actually captured*
  candidate silently ran eager instead of graph-replayed. PR example:
  `max_batch_size=1`, `max_seq_len=128` — ISL 64 ran piecewise, but ISLs
  100/107/121/127 all padded to 128 (no graph) and dropped to eager.
- **Root cause:** The model-engine filter on the piecewise capture candidate
  list used `i <= max_num_tokens`, looser than the engine's real reachable
  ceiling `max_batch_size * (max_seq_len - 1 - num_extra_decoding_steps)`.
  Entries above the ceiling were advertised as captured but warmup silently
  failed to record them; the padding logic then padded prefill chunks to a
  target with no captured graph and fell back to eager.
- **How introduced:** incomplete coverage of the capture-set bound; the PR
  names no culprit commit. Its before/after table shows the shape: with a
  power-of-two candidate ladder and `max_seq_len=128`, the top candidate
  (128) is unreachable, so every ISL in (64, 128) pads to a size with no
  graph and runs eager. The capture list was not derived from the reachable
  shapes, so the ceiling entry those ISLs needed never existed. A user-side
  workaround follows from the mechanism: choose `max_seq_len` so the
  ladder's top entry is reachable (a power of two plus 1).
- **Fix mechanism:** New `_filter_piecewise_capture_num_tokens` helper in
  `tensorrt_llm/_torch/pyexecutor/model_engine.py` (a) caps candidates at the
  reachable ceiling and (b) appends the ceiling itself so ISLs in the gap
  between the next-largest candidate and the ceiling still hit a graph;
  dropped entries are reported via a `logger.warning` pointing at
  `max_seq_len` (previously silent). **Half (b) did not survive:** appending
  the ceiling unconditionally made a far ceiling (`max_seq_len=65536` against a
  user list topping out at 13914) capture a huge extra graph nobody replays,
  which is nvbug `6404567` — PR #16256 replaced the append with a clamp of the
  largest user entry, see
  [piecewise CUDA-graph far-ceiling append](piecewise-cudagraph-far-ceiling-append.md).
  Cite (a) as the durable half of this fix; cite (b) only as the step that was
  itself rolled back.
- **Detection signal:** Prefill at ISLs just below `max_seq_len` shows eager
  kernel launches (no graph replay) in nsys while smaller ISLs replay graphs;
  post-fix builds log the drop —
  `grep "Skipping piecewise CUDA graph capture" <serve log>`. Also compare
  the configured `capture_num_tokens` list against
  `max_batch_size * (max_seq_len - 1)`: any advertised entry above that
  ceiling was never capturable.
- **Prevention/guard:** PR #13574 added the warning log plus unit tests
  (`tests/unittest/llmapi/test_llm_args.py::TestPiecewiseCudaGraphCaptureDefaults`)
  pinning the filtered capture set. General guard: warmup must fail loudly
  (or re-derive the set) whenever an advertised capture candidate cannot be
  recorded — advertised-but-uncaptured entries are exactly the silent-eager
  window.
- **Generalizes to:** `pattern-fast-path-silent-fallback` — a fast path's
  eligibility set is computed from a looser bound than the runtime's real
  reachable set, so boundary shapes silently take the slow path. Carries to:
  decode CUDA-graph batch-size lists that omit the max reachable batch;
  torch.compile / autotuned-shape caches whose bucket list misses a reachable
  shape (recompile or eager at runtime); spec-decode capture sets that ignore
  extra draft tokens when computing the token ceiling; any padding logic that
  pads toward a target the warmup never materialized.
