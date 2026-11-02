---
id: case-flinear-bias-copy-sm100
type: regression-case
family: kernel-and-fusion
module: kernel-fusion
maturity: full
regression_class: [fast-path-fallback]
signals: [throughput-drop, slower-kernel-in-trace]
subsystems: [model-definition, gemm-kernel]
introduced_via: [incomplete-coverage]
phase: [any-phase]
patterns: [pattern-fast-path-silent-fallback]
nvbugs: ["5914691"]
commits: ["b103625ebff8"]
success_prs: [11668]
failed_prs: []
---

# GPT-OSS linears hit F.linear's extra bias memory copy on SM100+

> Part of the [Kernel fusion regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `5914691` · commit `b103625ebff8` · PR #11668 —
  "[fix] WAR F.linear perf regression for GPTOSS".
- **Symptom:** perf regression for the GPT-OSS model on SM 100+ GPUs
  (per the fix's title and in-code comment). The PR states no number. The
  layers routed by the gate are GPT-OSS's biased linears (`qkv_proj`,
  `o_proj`).
- **Root cause:** on SM 100+, `F.linear` performs an additional memory copy
  for biases (per the diff comment). GPT-OSS's custom cuBLAS MM path
  (`torch.ops.trtllm.cublas_mm`, which takes the bias directly in
  `linear.py`'s `apply`) bypasses that copy, but its gate in
  `modeling_gpt_oss.py` was `sm_version == 121`, so all other SM 100+
  parts silently ran the slower default `F.linear` path.
- **How introduced:** incomplete arch coverage of a pre-existing fast path —
  the custom cuBLAS gate was written as an exact `sm_version == 121` match.
  The fix PR names no regressing commit.
- **Fix mechanism:** workaround (WAR): widen the gate in
  `Transformer.__init__` to `self.use_custom_cublas_mm = sm_version >= 100`,
  routing GPT-OSS linears through `torch.ops.trtllm.cublas_mm` on all
  SM 100+ GPUs. The underlying `F.linear` bias-copy cost is routed around,
  not removed.
- **Detection signal:** nsys trace of a GPT-OSS linear shows an extra copy
  kernel adjacent to each biased GEMM on SM 100+ — a separate bias add/copy
  whose byte count matches the bias shape on the `F.linear` path, absent once
  `use_custom_cublas_mm=True`. Check the gate with
  `grep -n "use_custom_cublas_mm = " tensorrt_llm/_torch/models/modeling_gpt_oss.py`
  and compare against the machine's SM version.
- **Prevention/guard:** none added by the PR (a 2-line gate change). Gap:
  arch-gated fast paths should use capability ranges (`>=`) rather than
  exact SM equality, and each supported arch needs a perf bar that would
  catch the default path running.
- **Generalizes to:** `pattern-fast-path-silent-fallback` — a fast path is
  silently disabled and a correct-but-slow generic path runs. Carries to:
  any exact `sm_version ==` gate that excludes newer/adjacent arches; other
  models whose linears default to `F.linear` and inherit its hidden bias
  copy on SM 100+; per-arch tuned kernel selections (LUTs) not extended
  when new hardware ships; dtype/shape-gated custom ops that quietly fall
  back to the framework op outside their gate.
