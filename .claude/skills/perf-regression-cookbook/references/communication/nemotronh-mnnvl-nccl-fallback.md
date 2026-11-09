---
id: case-nemotronh-mnnvl-nccl-fallback
type: regression-case
family: communication
module: communication
maturity: full
regression_class: [fast-path-fallback, communication-regression]
signals: [slower-kernel-in-trace, throughput-drop, itl-increase]
subsystems: [model-definition, communication]
introduced_via: [incomplete-coverage]
phase: [any-phase]
patterns: [pattern-fast-path-silent-fallback]
nvbugs: ["6264844"]
commits: ["0ff7b4acba23"]
success_prs: [15294]
failed_prs: []
---

# Nemotron-H allreduce silently falls back from MNNVL to NCCL on NVL multi-node

> Part of the [Communication regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6264844` · commit `0ff7b4acba23` · PR #15294 —
  Fix wrong NCCL fallback in nemotron-h.
- **Symptom:** On NVL multi-node deployments of Nemotron-H, every allreduce
  runs the NCCL path instead of the MNNVL fast path, "with worse perf"
  (PR #15294; no quantitative delta stated). Surfaced via nvbug 6264844.
- **Root cause:** `AllReduce.__init__`
  (`tensorrt_llm/_torch/distributed/ops.py`) only constructs its
  `MNNVLAllReduce` backend when a `dtype` is passed — the guard is literally
  `MNNVLAllReduce(self.mapping, dtype) if dtype else None`, and
  `MNNVLAllReduce.is_mnnvl(mapping, dtype)` also fails for `dtype=None`.
  All three `AllReduce` constructions in
  `tensorrt_llm/_torch/models/modeling_nemotron_h.py` (`NemotronHMOE`,
  `NemotronHLayer.pre_allreduce`, `NemotronHModel.final_allreduce`) omitted
  `dtype`, so MNNVL setup was silently skipped and the op fell back to NCCL.
- **How introduced:** unknown — the PR names no regressing commit. The diff
  shows an enablement gap: the MNNVL fast path's dtype-at-construction
  requirement was not covered by this model's `AllReduce` call sites.
- **Fix mechanism:** Passes `dtype=config.torch_dtype` to all three
  `AllReduce` constructions in `modeling_nemotron_h.py`, plus a comment at
  each site: "AllReduce needs dtype at construction to build fused MNNVL
  paths."
- **Detection signal:** NCCL allreduce kernels in the trace where MNNVL
  kernels are expected on NVL multi-node; at debug log level the constructor
  emits "MNNVL AllReduce can't be enabled due to ..." / "failing the
  is_mnnvl check". Audit call sites with
  `grep -n "AllReduce(" tensorrt_llm/_torch/models/modeling_nemotron_h.py`
  and confirm each passes `dtype=`.
- **Prevention/guard:** The fix adds only source comments; the fallback
  remains a `logger.debug`, invisible at default log level (gap). A
  warning-level log (or assert under `strategy=MNNVL`) when `dtype is None`
  disables MNNVL, or a lint that model-level `AllReduce(...)` constructions
  pass `dtype`, would catch this class at bring-up.
- **Generalizes to:** pattern-fast-path-silent-fallback — an optional
  constructor arg gates a fast path, and omitting it degrades silently;
  carries to any other model file constructing `AllReduce` without `dtype`,
  `SYMM_MEM` allreduce (defaults missing dtype to bfloat16 rather than the
  model dtype), new-model onboarding that copies pre-MNNVL constructor
  signatures, and any backend whose only fallback breadcrumb is a
  debug-level log.
