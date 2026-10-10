---
id: case-maybe-compile-recompiles-every-call
type: regression-case
family: kernel-and-fusion
module: jit-and-warmup
maturity: full
regression_class: [host-work-added]
signals: [host-time-increase, throughput-drop, itl-increase]
subsystems: [attention-kernel, runtime-python]
introduced_via: [prior-fix-side-effect]
phase: [any-phase]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["5631254", "5631229", "5650079"]
commits: ["8bd779171e99"]
success_prs: [9135]
failed_prs: []
---

# maybe_compile called torch.compile inside the wrapper — every invocation re-entered the compiler

> Part of the [JIT & warmup regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbugs `5631254`, `5631229`, `5650079` · commit `8bd779171e99` ·
  PR #9135 — "[https://nvbugs/5631254][fix] avoid torch.compile for multiple
  times". Three bugs, one three-line root cause, one PR — so one case: a
  DeepSeek-R1 FP4 performance regression (5631254), the same on FP8 (5631229),
  and a DeepSeek-R1 performance drop on `main` attributed to torch.compile ops
  (5650079).
- **Symptom:** DeepSeek-R1 throughput drops on both FP4 and FP8. **The PR body is
  empty and states no symptom, no number, no model config and no hardware** — the
  magnitude has to come from the three bugs, and 5650079 is the one that names
  the mechanism outright: a performance drop caused by torch.compile ops.
- **Root cause:** the `maybe_compile` decorator added in PR #8708 built its
  compiled artifact **inside** the wrapper body:

  ```python
  def wrapper(*args, **kwargs):
      if <piecewise not running>:
          return torch.compile(f, **compile_kwargs)(*args, **kwargs)
  ```

  `torch.compile(f, ...)` returns a *new* `OptimizedModule` on every call, so the
  guard/dispatch machinery was re-created per invocation instead of once per
  decorated function. The compiled code itself is cached by torch, so this is not
  a full recompile every call — it is per-call wrapper construction and guard
  re-installation, which is pure host time on the hot path. At the fix commit the
  affected call sites are `tensorrt_llm/_torch/modules/attention.py` L79-85
  (invoked from L1324, L1439, L1534, L1593), `layer_norm.py:70`, and
  `sparse/dsa.py:671` — i.e. MLA attention, layer norm, and the DSA sparse path,
  which is why one three-line defect moved DeepSeek-R1 in both precisions.
- **How introduced:** `prior-fix-side-effect`. `maybe_compile` exists only because
  PR #8708 needed to make the compile conditional (see
  `case-piecewise-attention-torch-compile-host-overhead`); the conditional was
  written as "decide, then compile, then call", and moving the decision inside the
  wrapper silently moved the compilation there too.
- **Fix mechanism:** hoist the compilation to **decoration** time — build
  `compiled_func = torch.compile(f, **compile_kwargs)` once in the decorator body
  and have the wrapper choose between `compiled_func(...)` and `f(...)`. One file,
  three lines changed. The conditional behaviour of #8708 is preserved exactly;
  only the construction moves.
- **Detection signal:** static and unambiguous —
  `git grep -n "return torch.compile(f, \*\*compile_kwargs)(\*args, \*\*kwargs)"`
  matches only a pre-fix tree. Generalize the probe: any `torch.compile(` (or
  `functools.lru_cache`, `torch.jit.script`, …) whose call appears *inside* a
  function that runs per request/step, rather than at module or decoration scope,
  is the same defect. In a profile the signature is a flat per-iteration host
  overhead across several unrelated modules at once (attention + norm + sparse),
  which is what points at a shared decorator rather than at any one kernel.
- **Prevention/guard:** **no test was added** — the three-line fix ships alone.
  The gap worth naming: a decorator that composes a compiler call with a runtime
  predicate has two very different correct shapes, and both type-check and both
  produce right answers. A unit test asserting the decorated function's compiled
  artifact is *identical across two calls* (`f2 = deco(g); assert f2(...) is
  ... ` / capture `torch.compile` with a counting mock and assert it is invoked
  once per decoration) would have caught it; nothing in the suite does.
- **Generalizes to:** `pattern-host-work-on-hot-path`; carries to every
  compile/cache/JIT wrapper introduced to make an optimization *conditional* —
  the conditional is the moment the construction tends to slip into the call
  path. Read with `case-piecewise-attention-torch-compile-host-overhead` (the PR
  that created this helper) and
  `case-mla-chunked-prefill-maybe-compiled-cat-warmup` (the warmup gap on the same
  op): one helper, three separate performance bugs, landing October →
  November 2025 → March 2026.
