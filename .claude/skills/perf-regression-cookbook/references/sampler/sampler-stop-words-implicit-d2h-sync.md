---
id: case-sampler-stop-words-implicit-d2h-sync
type: regression-case
family: execution-and-graph
module: sampler
maturity: full
regression_class: [sync-introduced]
signals: [gpu-idle-between-steps, throughput-drop, itl-increase]
subsystems: [sampler, runtime-python]
introduced_via: [new-feature]
phase: [decode]
patterns: [pattern-per-step-sync-added]
nvbugs: ["5738737"]
commits: ["696f754ef455"]
success_prs: [10120]
failed_prs: []
---

# torch.equal in stop-words checking forced a per-request D2H sync inside sample_async

> Part of the [Sampler regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5738737` · commit `696f754ef455` · PR #10120 —
  "[None][fix] avoid implicit cudaStreamSynchronize in sample_async." **The PR
  names no bug**; the link comes from the bug, which names PR #10120 repeatedly
  and attributes the root cause to a CUDA stream synchronization in the sampler.
- **Symptom:** a Nemotron Nano v3 performance regression in online serving —
  roughly a **3× drop** in output token throughput measured through
  `trtllm-serve` with aiperf and
  `python -m tensorrt_llm.serve.scripts.benchmark_serving`: **10,607.09 →
  4,433.11 tokens/sec**. The bug names the culprit commit directly:
  `02edb19f4302` (stop-words support), also filed upstream as
  NVIDIA/TensorRT-LLM issue #9965. The PR's own justification is structural rather
  than numeric — "Any cudaStreamSynchronize in `sample_async` will break the
  overlap feature of overlap scheduler and pipeline parallelism" — with nsys
  screenshots for "LLaMA 405B TP2PP2" and the author's explicit caveat "No e2e
  perf data".
- **Root cause:** the stop-words check `_are_stop_words` compared a candidate
  token window against each stop-word with `torch.equal(...)`. `torch.equal`
  returns a Python `bool`, so it must synchronize — **once per request, per beam,
  per step**. Two supporting costs in the same function: `lens_device` was
  iterated in Python (another D2H), and a Python list was used to index CUDA
  tensors. Because the sync sits inside `sample_async`, it does not merely cost
  its own latency: it collapses the overlap scheduler's pipelining and PP's
  send/recv overlap, which is how a per-request check becomes a 3× throughput
  loss instead of a few microseconds.
- **How introduced:** `new-feature` — stop-words support (`02edb19f4302`). The
  feature is correct; the comparison idiom is what syncs. Note the resolution had
  two halves: the culprit PR was **reverted** in PR #10002 and functional
  stop-token support later re-landed in PR #10389, while #10120 is the PR that
  removed the sync (per the NVBug). Cite #10120 for the mechanism.
- **Fix mechanism:** make the whole check device-resident. Stop-word lengths move
  through **pinned int32** memory with `non_blocking=True`; the comparison becomes
  an accumulation of `(truncated_seq == word[:L]).all()` on device; the Python
  list indexing is removed. `sampler.py` is +20/−19. Crucially the **early exit is
  deliberately removed** — an early exit would require reading the result on the
  host, which is the sync being deleted. That trade is documented in the test:
  `test_write_finish_reasons`' expectation changes to
  `[STOP_WORDS, NOT_FINISHED, STOP_WORDS]` with the comment "We don't use early
  exit to avoid stream synchronization for stop words". So a future reader who
  "optimizes" by restoring the short-circuit re-introduces the regression, and the
  test tells them so.
- **Detection signal:** `git grep -n "torch.equal" tensorrt_llm/_torch/pyexecutor/sampler.py`
  — any match is a candidate defect. Generalize: `torch.equal`, `.item()`,
  `bool(tensor)`, `if tensor:`, and iterating a CUDA tensor are all implicit
  syncs, and the ones that matter are the ones inside `sample_async` /
  post-processing. In a profile, `sample_async` is NVTX-annotated: a
  `cudaStreamSynchronize` inside that range, plus GPU idle whose duration scales
  with **batch size × number of stop words** rather than with tokens, is the
  signature.
- **Prevention/guard:** no new test guards the *absence* of a sync — the existing
  test only pins the behavioural consequence of dropping the early exit. The rule:
  any host-visible boolean derived from device data on the sampling path is a
  design error, not a micro-optimization opportunity. Overlap-sensitive regions
  deserve an explicit "no host reads" comment (this PR adds one) because the
  offending idioms look like ordinary Python.
- **Generalizes to:** `pattern-per-step-sync-added`; carries to every per-request
  predicate evaluated on device data (stop words, max-length checks, finish
  reasons, guided-decoding masks), to `torch.equal`-style comparisons anywhere on
  the step path, and to the general result that **overlap-scheduler and PP
  configurations amplify a single sync into a multiple-× throughput loss** — the
  same defect on a non-overlapped single-GPU run would have read as noise.
