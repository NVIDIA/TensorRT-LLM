---
id: case-fp8-pertensor-quant-slower-than-te
type: regression-case
family: kernel-and-fusion
module: gemm-and-quantization
maturity: full
regression_class: [kernel-selection-regression]
signals: [slower-kernel-in-trace, throughput-drop]
subsystems: [gemm-kernel, autotuner]
introduced_via: [pre-existing-gap]
phase: [any-phase]
patterns: [pattern-kernel-swap-regressed]
nvbugs: ["5846489"]
commits: ["f39e1a8603f9"]
success_prs: [11057]
failed_prs: []
---

# Native FP8 per-tensor quantization kernel slower than TE's on H100

> Part of the [GEMM & quantization regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5846489` · commit `f39e1a8603f9` · PR #11057 —
  Apply TE's FP8 per-tensor quantization.
- **Symptom:** "TRT-LLM's FP8 per-tensor quantization performs slower than
  TE's on H100" (PR #11057). This is a kernel-level comparison, not an e2e
  number: the PR states no magnitude and no end-to-end delta.
- **Root cause:** the native
  `torch.ops.tensorrt_llm.quantize_e4m3_per_tensor` kernel was
  unconditionally used for FP8 E4M3 per-tensor activation quantization even
  though TE's `Float8CurrentScalingQuantizer` kernel is faster on H100 —
  only one implementation existed, so the slower one always ran.
- **How introduced:** unknown — not stated in the PR; below-alternative
  performance of the existing kernel, not a regression from a previously
  faster TRT-LLM state.
- **Fix mechanism:** adds a `QuantizeE4M3PerTensorRunner(TunableRunner)` in
  `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` exposing two tactics
  ("trtllm" native kernel, "te" via `transformer_engine_torch`) behind a new
  `trtllm::quantize_e4m3_per_tensor` custom op; `AutoTuner.choose_one`
  profiles both per num-tokens bucket (up to `tune_max_num_tokens=8192`) and
  caches the faster backend. TE import is lazy; when TE is missing it logs a
  warning and keeps the native tactic only.
- **Detection signal:** nsys trace shows the quantize kernel preceding FP8
  GEMMs taking longer than the TE equivalent for the same shape; verify the
  autotuned op exists and TE was picked up:
  `grep -n "QuantizeE4M3PerTensorRunner" tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`
  and run the PR's unit test (`pytest -v -k
  "test_quantization_dequantization_per_tensor or
  test_quantization_per_tensor_scales"
  tests/unittest/trt/quantization/test_fp8_quantization.py`) or invoke the op
  directly to see whether TE is importable — the
  `Transformer Engine not available` warning is emitted only in contexts that
  actually invoke `trtllm::quantize_e4m3_per_tensor`, so it does not appear
  in serve logs.
- **Prevention/guard:** the fix supplies the guard mechanism — autotune-based
  backend selection instead of a hardcoded kernel — but the PR does not wire
  it into any production quantization call site (call sites still use the
  native `torch.ops.tensorrt_llm.quantize_e4m3_per_tensor` ops as of the fix
  commit), so the guard only takes effect once callers adopt the new
  `trtllm::quantize_e4m3_per_tensor` op; PR adds unit tests
  (`tests/unittest/trt/quantization/test_fp8_quantization.py`,
  `test_quantization_dequantization_per_tensor`) covering the autotuned op.
  Gap: correctness tests only — no perf bar asserts the faster backend is
  actually chosen.
- **Generalizes to:** pattern-kernel-swap-regressed — a single hardcoded
  kernel implementation loses to an available alternative on some
  arch/shape; keep both selectable and let measurement decide. Recurs for
  any elementwise/quantize op with a library twin (TE, cuBLASLt epilogues,
  torch native), for per-block/per-token FP8 or FP4 quant variants with only
  one backend wired, and when a hand-rolled kernel is kept after a faster
  vendor kernel becomes importable in the container.
