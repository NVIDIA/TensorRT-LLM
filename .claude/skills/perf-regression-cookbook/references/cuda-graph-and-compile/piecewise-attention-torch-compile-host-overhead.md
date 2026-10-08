---
id: case-piecewise-attention-torch-compile-host-overhead
type: regression-case
family: kernel-and-fusion
module: cuda-graph-and-compile
maturity: full
regression_class: [host-work-added]
signals: [ttft-increase, host-time-increase, throughput-drop]
subsystems: [attention-kernel, cuda-graph, runtime-python]
introduced_via: [new-feature]
phase: [prefill]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["5550409", "5948590"]
commits: ["a69bd2a6fab9"]
success_prs: [8708]
failed_prs: []
---

# torch.compile on small MLA context ops costs host time inside piecewise CUDA graphs

> Part of the [CUDA graph & compile regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbugs `5550409`, `5948590` · commit `a69bd2a6fab9` · PR #8708 —
  "[https://nvbugs/5550409][fix] Disable torch compile in piecewise attention
  part to Avoid host overhead". Two bugs, one root cause, one PR — so one case:
  5550409 reports a perf regression with torch.compile + piecewise CUDA graph
  between main and 1.1rc5 on GB200; 5948590 reports `deepseek_v3_lite`
  `Inference_Time` ~10% worse on release/1.2 than on 1.1.0, on RTX 6000 SE.
- **Symptom:** with piecewise CUDA graph enabled, the MLA context path spends
  extra **host** time and TTFT rises; the RTX 6000 SE bug reports ~10 %
  `Inference_Time` against 1.1.0 on `deepseek_v3_lite`. The PR body states the
  reasoning rather than a number: the piecewise-attention region covers "small
  ops" whose cost is host-side, so "it has a high possibility that they are
  host-bound cases. Thus, disabling it would improve TTFT". **The PR quotes no
  percentage, model config or hardware** — take the magnitude from the two bugs,
  not from the PR.
- **Root cause:** `compiled_copy_` and `compiled_cat` in
  `tensorrt_llm/_torch/modules/attention.py` were bare `@torch.compile`
  functions, called from the MLA context paths (`forward_context_default`,
  `forward_context_with_cached_kv`, `forward_context_with_chunked_prefill`).
  Inside a *piecewise* CUDA graph the surrounding region is already replayed as
  a graph, so what remains around these ops is pure host work — and a compiled
  op's guard evaluation plus dispatch is host work that a plain eager `cat` /
  `copy_` does not pay. The ops are small enough that the compile buys nothing
  to offset it. So the cost is not "the kernel got slower"; it is dispatch
  overhead landing on a path whose budget is host-bound by construction.
- **How introduced:** the piecewise-CUDA-graph execution mode is what makes the
  compiled wrappers a net loss; the wrappers themselves predate it. Both bugs
  are branch-comparison regressions (main vs 1.1rc5; release/1.2 vs 1.1.0), i.e.
  the cost appeared when the piecewise path became the executed one, not when
  the ops were written.
- **Fix mechanism:** make the compile **conditional on not running piecewise**.
  PR #8708 adds an `is_piecewise_running_flag` plus a `maybe_compile` helper in
  `tensorrt_llm/_torch/utils.py`, renames the two ops to `maybe_compiled_copy_` /
  `maybe_compiled_cat`, and threads a `piecewise_runner_num` count through the
  attention modules so the decision can be made per configuration. Two quirks a
  re-implementation must preserve, both non-obvious from the names:
  the single-runner configuration sets the flag **False** (one runner is not the
  case this guards), and the flag assignment sits *after* the
  `get_piecewise_cuda_graph_flag()` early return, so a caller that returns early
  never sets it.
- **Detection signal:** static — `python -c "import tensorrt_llm._torch.modules.attention as a;
  print(hasattr(a,'compiled_cat'), hasattr(a,'maybe_compiled_cat'))"`: `(True,
  False)` is a pre-fix tree, `(False, True)` is post-fix. In a profile, the
  signature is host span growth in the MLA context region with **no** change to
  kernel names or device time, and it appears only when piecewise CUDA graph is
  on — so A/B the piecewise flag before bisecting kernels.
- **Prevention/guard:** **no test and no guard were added.** The transferable
  rule: `torch.compile` on a *small* op is only a win when its dispatch cost is
  amortized by the work it fuses, and inside a piecewise/graph-replayed region
  there is no such amortization. Any `@torch.compile` on a function reachable
  from a graph-captured region should be gated on the execution mode, not
  applied unconditionally at decoration time.
- **Generalizes to:** `pattern-host-work-on-hot-path`. Read together with the two
  neighbouring cases on the same machinery — `case-maybe-compile-recompiles-every-call`
  (PR #9135, the follow-up defect *inside* the `maybe_compile` helper this PR
  introduced) and `case-mla-chunked-prefill-maybe-compiled-cat-warmup`
  (PR #11743/#11744, the warmup gap on the same `maybe_compiled_cat`). Three
  distinct root causes on one helper, in landing order — when auditing a
  `maybe_compile`-style wrapper, check all three: is it compiled at all, is it
  compiled once, and is the compile paid at warmup?
