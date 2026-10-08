---
id: case-qwen35-fused-ops-disabled-under-torch-compile
type: regression-case
family: kernel-and-fusion
module: kernel-fusion
maturity: full
regression_class: [fast-path-fallback]
signals: [ttft-increase, itl-increase, many-small-kernels]
subsystems: [attention-kernel, cuda-graph, model-definition]
introduced_via: [incomplete-coverage]
phase: [any-phase]
patterns: [pattern-fast-path-silent-fallback]
nvbugs: ["6545424"]
commits: ["63eb09509706"]
success_prs: [17243]
failed_prs: []
---

# Qwen3.5 fused QK-norm/RoPE/gate ops self-disabled under torch.compile because raw Triton was Dynamo-visible

> Part of the [Kernel fusion regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6545424` · commit `63eb09509706` · PR #17243 — registers
  the fused ops as custom ops and deletes their `is_torch_compiling()` guards.
- **Symptom:** the NVBug carries the number: with piecewise graphs, ITL and
  TTFT rise by **5–20 %** on Qwen3.5-4B-FP8. The PR states the same effect
  qualitatively — "These fallbacks increased piecewise CUDA graph TTFT for
  long-context serving" — and reports "42 focused H100 tests passed".
- **Root cause:** three fused paths — fused QK-norm + RoPE + gate, an in-place
  sigmoid-mul, and a token-major gated RMSNorm — were implemented as **raw Triton
  launchers**, i.e. plain Python functions Dynamo can see into. Under
  `torch.compile` Dynamo would trace through them and either graph-break or
  mis-handle the in-place mutation, so the code protected itself with
  `if is_torch_compiling(): <eager fallback>`. The result is the worst combination:
  with piecewise CUDA graphs enabled (the configuration you turn on *for* latency)
  the fused kernels are exactly the ones that switch themselves off, and the model
  runs the unfused sequence of small ops. Nothing errors and nothing logs, so the
  configuration reads as "piecewise is slower on this model".
- **How introduced:** `incomplete-coverage` — the fused kernels shipped without a
  `torch.library` registration, and the guard was the honest short-term
  accommodation. It became a perf defect when piecewise CUDA graphs became the
  default operating mode for this model class.
- **Fix mechanism:** make the kernels opaque to Dynamo instead of avoiding it —
  register each as a custom op with a fake/meta implementation:
  `trtllm::fused_qkv_gemma_rmsnorm_rope_gate` (with `register_fake`; the mrope
  tuple argument is flattened into `use_mrope` / `mrope_section1` /
  `mrope_section2` because custom-op schemas cannot carry nested tuples),
  `trtllm::fused_sigmoid_mul_inplace` (declared `mutates_args=("attention_output",)`
  — the mutation is *declared*, not hidden), and
  `trtllm::rms_norm_gated_token_major`. Then **both** `is_torch_compiling()` guards
  are deleted, 11 `optional_inplace_infos` entries are added so the piecewise
  pass understands the in-place ops, and a `SAVE_RSTD` heuristic is added. The
  schema-flattening detail is the reusable part: a fused kernel that cannot be
  registered as-is usually needs its *signature* changed, not a guard.
- **Detection signal:** `grep -n "is_torch_compiling" tensorrt_llm/_torch/modules/qk_norm_attention.py`
  — any match is a self-disabling fast path. Generalise:
  `git grep -n "is_torch_compiling" tensorrt_llm/` and, per hit, ask what the
  *else* branch costs when compile is on. In a profile with piecewise enabled, the
  signature is many small elementwise/norm kernels where a single fused kernel
  appears in the compile-off run — so **A/B piecewise on and off** and treat "faster
  with piecewise off" as evidence of a fallback, not of graph overhead.
- **Prevention/guard:** the new tests are **not mapped into `test_lists/`**, so CI
  does not run them — the guard exists in the tree but not in the pipeline. Worth
  stating plainly, because the same class recurs: no test asserts that the fused op
  is *reached* under compile. The durable rule: a kernel must never be gated on "am
  I being compiled?"; if Dynamo cannot handle it, register it as a custom op so
  Dynamo does not need to.
- **Generalizes to:** `pattern-fast-path-silent-fallback`; carries to every
  `is_torch_compiling()` / `torch._dynamo.is_compiling()` guard, to raw Triton
  launchers on a compiled path, and to in-place fused ops whose mutation is
  undeclared. Related in this family: the `maybe_compile` cases, which are the
  mirror image — there the compile decision was made *per call* rather than
  disabled.
