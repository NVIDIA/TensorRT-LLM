---
id: case-custom-op-dispatch-tax-ltx2
type: regression-case
family: execution-and-graph
module: runtime-and-serving
maturity: full
regression_class: [host-work-added]
signals: [host-time-increase, gpu-idle-between-steps, throughput-drop]
subsystems: [runtime-python]
introduced_via: [new-feature]
phase: [any-phase]
patterns: [pattern-host-work-on-hot-path]
nvbugs: ["6081154"]
commits: ["95db86891de3"]
success_prs: [13149]
failed_prs: []
---

# `@torch.library.custom_op` dispatch tax on the LTX-2 NVFP4 quantize hot path

> Part of the [Runtime & serving regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6081154` (a visual-gen performance bug: LTX2
  regressed at f7782cb75 against 5653803a53; closed as fixed and verified) ·
  commit `95db86891de3` · PR #13149 — "[TRTLLM-11958][perf] reduce
  @torch.library.custom_op host overhead". Note the PR carries **no NVBug id
  in its title or body**: the linkage is that, per the NVBug, nsys profiling
  named `tunable_fp4_quantize`'s host dispatch overhead as the root cause, and
  PR #13149 migrates exactly that op (plus `nvfp4_gemm`) off `@custom_op`. The
  bug records only an interim workaround (enabling CUDA graph); #13149 is the
  code fix that removed the underlying tax.
- **Symptom:** LTX-2 NVFP4 image-to-video E2E latency up **+19.5 %** — 9.95 s
  → 11.89 s for 768×1280, 121 frames, 40 steps, cfg=2, ulysses=4 on 8× B200
  (numbers per the NVBug). The regression appears **only at 8 GPU without
  CUDA graph**: 1 GPU showed none (the new build was 2.7 % faster), and with
  `enable_cuda_graph: true` it vanished entirely (9.78 s new vs 9.82 s old).
  Surfaced on the `ltx2-i2v-sfp4-vanilla-2x4-cache0-tcompile1-cg1` benchmark
  row.
- **Root cause:** `@torch.library.custom_op` wraps every call in Python-level
  kernels registered on several dispatch keys — an Autograd kernel
  (`is_grad_enabled()` + `_any_requires_grad()` tensor iteration, `Metadata`
  construction, an `_AutoDispatchBelowAutograd` context manager, and a second
  `op.redispatch()` trip) plus a CUDA `backend_impl` wrapper that runs
  `_c_check_aliasing_constraint()` on every call. PR #13149 measures the total
  as a **~7 µs per-call dispatcher tax** (B200 + PyTorch 2.10, 20k-iter loop:
  plain Python fn 4.72 µs vs `@custom_op` 11.67 µs). LTX-2's dense transformer
  issues **~2100 calls per step** across the two ops (~1260
  `tunable_fp4_quantize` + ~840 `nvfp4_gemm`), so the wrapper alone costs
  **~11.3 ms/step**. Because that cost is host-side and **non-uniform across
  the 8 workers**, 2 of 8 dropped from 99 % to 60 % GPU utilization (2200+
  inter-kernel gaps >10 µs, ~100 ms/step of idle); those stragglers arrived
  late at every NCCL collective and the other ranks spin-waited inside the
  NCCL kernels, so NCCL time per step doubled (69 ms → 137 ms) — 40 steps ×
  ~67 ms ≈ the ~2.7 s observed.
- **How introduced:** PR #12126 (commit `1480140211f1`,
  "[TRTLLM-11091][feat] Add tunable nvfp4 quantize with additional FlashInfer
  backend"), identified by the bug's bisect over 46 commits. It set
  `NVFP4LinearMethod.use_tunable_quantize = True` for visual gen, replacing
  the plain `fp4_quantize` call with the `tunable_fp4_quantize` **custom op** —
  a new feature paying the wrapper tax ~1260 times per step on a path that
  previously did not.
- **Fix mechanism:** Adds `fast_custom_op`
  (`tensorrt_llm/_torch/custom_ops/fast_custom_op.py`, new), a thin decorator
  that registers the op through the low-level `torch.library.Library.define +
  impl` API — `FRAGMENT` mode (the `trtllm` namespace is already declared from
  C++), schema still inferred from Python type hints via
  `torch.library.infer_schema`, and a `FastCustomOp` handle that keeps
  `.register_fake` and caches the `OpOverload`. No Autograd kernel, no
  ADInplaceOrView wrapper, no per-call aliasing check: the call path collapses
  to `C++ dispatcher → CUDA key → user_fn`. `trtllm::nvfp4_gemm` and
  `trtllm::tunable_fp4_quantize` are switched to `@fast_custom_op` in
  `torch_custom_ops.py` (**−5.39 µs/call**, 11.67 → 6.28 µs). Measured E2E:
  8 GPU 10.21 s → 9.67 s (−5.3 %), 1 GPU 33.95 s → 33.61 s (−1.0 %), nsys
  pure CPU idle −29 ms/step. Both ops qualify because they are pure
  (`mutates_args=()`) and inference-only — the op keeps the identical
  `OpOverload` and schema, so `torch.compile`, Dynamo and FX pattern matchers
  (e.g. `ar_residual_norm` matching `torch.ops.trtllm.nvfp4_gemm.default`) are
  unaffected.
- **Detection signal:** GPU kernel inventory is **unchanged** while wall time
  grows — the bug's nsys diff found identical kernel counts (5478 = 5091
  compute + 387 NCCL), identical compute GPU time (131 ms/step on every
  device) and identical NCCL grid sizes at 2× duration, with 99 % of the idle
  in compute↔compute transitions rather than at collective boundaries. Two
  cheap discriminators: (1) flip `enable_cuda_graph: true` — if the gap
  closes, the cost is per-kernel CPU dispatch, not GPU work; (2) check
  **per-rank** GPU utilization, not the aggregate, since host jitter on 2 of 8
  ranks masquerades as a communication regression. To audit the wrapper tax
  itself: `grep -rn "@torch.library.custom_op" tensorrt_llm/_torch/custom_ops/`
  and cross-check any hit against its per-step call count — anything pure and
  inference-only in the thousands-of-calls range is a `fast_custom_op`
  candidate.
- **Prevention/guard:** None committed — the diff is two files
  (`fast_custom_op.py` new, 118 lines; `torch_custom_ops.py` +3/−2) and adds
  **no test**;
  the PR's 13 parity checks and its 10-step LTX-2 smoke run were ad-hoc
  validation, and existing coverage
  (`tests/unittest/_torch/thop/test_nvfp4_gemm.py`) passes unchanged because
  the op identity is preserved. The durable guard is the PR's own "When to
  keep `@custom_op`" decision table plus the caveats in
  `fast_custom_op.py`'s module docstring (no autograd, `mutates_args` must be
  a concrete tuple, `device_types` defaults to `"CUDA"`). Gaps: no lint or
  review check flags a wrapper-heavy op registration on a high-call-count
  path, and no visual-gen perf bar covers the 8-GPU non-cuda-graph config
  where this was the only place the regression was visible.
- **Generalizes to:** `pattern-host-work-on-hot-path` — a per-call framework
  wrapper, not the kernel, is the added host work, and it is charged once per
  op invocation on a path that issues thousands per step. Carries to: other
  `@torch.library.custom_op` ops still on per-step paths (only 2 were migrated
  here); features that flip a model family onto a wrapper-wrapped op variant
  (`use_tunable_quantize = True`) without counting call sites; host-overhead
  regressions that are **invisible at 1 GPU and only appear at scale**,
  because per-rank host jitter is amplified into doubled collective time by
  spin-wait — read per-rank utilization before blaming NCCL; and workloads
  where `torch.compile` + CUDA graph have already absorbed the other launch
  overhead, which makes a fixed per-call tax a much larger fraction of what
  remains (PR #13149 measures the two ops at ~2.5 % of per-step host time in
  eager, ~4.6 % under `torch.compile`, and ~13 % under `torch.compile` +
  cuda_graph).
