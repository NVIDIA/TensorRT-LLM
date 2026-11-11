---
id: case-mla-chunked-prefill-maybe-compiled-cat-warmup
type: regression-case
family: kernel-and-fusion
module: jit-and-warmup
maturity: full
regression_class: [warmup-jit-gap]
signals: [midrun-stall, itl-increase, perf-ci-bar-failure]
subsystems: [attention-kernel, cuda-graph]
introduced_via: [incomplete-coverage]
phase: [prefill]
patterns: [pattern-warmup-coverage-gap]
nvbugs: ["5823212"]
commits: ["85dc52acae7c", "6c542e98216d"]
success_prs: [11743, 11744]
failed_prs: []
---

# maybe_compiled_cat torch.compile stall on chunked-prefill MLA hot path

> Part of the [JIT & warmup regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5823212` · commit `85dc52acae7c` · PR #11743 —
  Warmup maybe_compiled_cat in forward_context_with_chunked_prefill.
  **Two-branch landing — cite both:** #11743 is the `main` PR (the commit
  above); PR #11744 is the byte-identical cherry-pick to `release/1.2`
  (commit `6c542e98216d`, same title, same two files, both merged
  2026-03-02). Both titles carry the same nvbug tag, so a PR→bug lookup that
  only follows `main` misses the `release/1.2` landing — cite both.
- **Classification note (why it lives here, not in the instability cookbook):**
  this case was filed under instability `warmup-and-jit` until 2026-08-12 and
  was moved. The mechanism is a deterministic first-use recompile, which is a
  reproducible deficit — not an outlier that sometimes lands in the measurement
  window. **The trap this case teaches: a deterministic first-use recompile
  presents *as* a first-iter spike while behaving as a regression.** The
  recompile fires the first time each stride pattern is seen in a process, so
  the per-iteration trace looks like classic JIT-on-hot-path instability; the
  discriminator is that every rep pays it identically, so the *mean* moves and
  the variance does not. Ask whether the cost is paid once per process
  (regression) or unpredictably (instability) before choosing a cookbook.
- **Symptom:** on MLA models with chunked prefill, the chunked-prefill MLA
  context path pays a `torch.compile` recompile whenever it meets a new input
  shape, adding host overhead to the affected iters — surfacing as a
  one-directional `gpu_time` mean shift on perf bars. The PR states no number.
- **Root cause:** the chunked-prefill MLA context path lets torch.compile
  compile `maybe_compiled_cat` at runtime. The obvious alternative — compiling
  with `dynamic=True` so one graph covers every shape — is ruled out by an
  in-code comment in the diff: "Do not use torch.compile with dynamic=True here
  because it completely ignores tensor layout/stride information, resulting in
  significantly degraded performance." So the fix's motive is to keep the op
  shape-specialized *and* move the compile out of the hot path; a future agent
  proposing `dynamic=True` here is re-proposing a known-bad option.
- **How introduced:** the compile-on-first-use behaviour was accepted for
  the MLA chunked-prefill path when it landed; no warmup exercised the op
  in the chunked-prefill call site.
- **Fix mechanism:** move the torch.compile compilation of
  `maybe_compiled_cat` out of the hot chunked-prefill MLA context path into
  an explicit cached warmup — a `@staticmethod @functools.cache`
  `cached_warmup_forward_context_with_chunked_prefill` in
  `tensorrt_llm/_torch/modules/attention.py`, plus a
  `is_chunked_prefill_mla_context_for_warmup` predicate in
  `attention_backend/trtllm.py` that is the served check *minus* the
  `num_ctx_cached_tokens > 0` term (warmup has no cached tokens, so the served
  predicate would never fire during warmup). The compile happens
  deterministically at warmup time and the served path finds a cached artifact.
  Two details from the diff that a re-implementation must keep: the warmup
  tensors are marked with `torch._dynamo.maybe_mark_dynamic(…, 0)` on the
  `num_tokens` dimension so **one** pass generalizes across all
  `num_tokens != 1`; and `num_tokens = 1` is warmed **separately** because
  torch.compile specializes for it and would otherwise still recompile inline.
  The covering argument, from the diff: the warmup exercises the two concat
  layouts the MLA path uses — `concat(chunked_k_nope, chunked_k_pe)` and
  `concat(k_nope, k_pe)` — at `num_tokens` 2 and 1, so each stride pattern is
  compiled before serving.
- **Detection signal:** an nsys torch-compile / inductor compile span
  visible in the first chunked-prefill iter's timeline (not the warmup
  region); `grep -n 'maybe_compiled_cat\|compile_fx' bench.log` and check
  whether the compile timestamp precedes or follows the warmup end marker.
- **Prevention/guard:** dynamic=True is a red flag on hot-path
  `torch.compile` calls when perf matters; every compiled op on a served
  path should have a cached warmup entry.
- **Generalizes to:** `pattern-warmup-coverage-gap`; carries to compiled ops
  living inside chunked-prefill / spec-decode branches, torch.compile calls
  wrapped inside per-shape helpers, and any inductor-compiled function
  reachable from an attention forward.
