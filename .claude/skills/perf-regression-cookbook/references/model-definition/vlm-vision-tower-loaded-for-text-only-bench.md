---
id: case-vlm-vision-tower-loaded-for-text-only-bench
type: regression-case
family: memory-and-capacity
module: model-definition
maturity: full
regression_class: [memory-footprint-regression]
signals: [kv-capacity-drop, memory-usage-increase, throughput-drop, perf-ci-bar-failure]
subsystems: [model-definition, runtime-python, kv-cache]
introduced_via: [new-feature]
phase: [any-phase]
patterns: [pattern-unaccounted-startup-residency]
nvbugs: ["6405760"]
commits: ["573bd5f0b841", "e8e1ade1c6e3"]
success_prs: [16250, 15632]
failed_prs: [15985]
---

# VLM checkpoint's vision tower loaded for text-only benchmarks, shrinking the KV pool

> Part of the [Model definition regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6405760` · commit `573bd5f0b841` · PR #16250 — "Do not
  load the multimodal encoder for text-only trtllm-bench runs"; related:
  commit `e8e1ade1c6e3` · PR #15632 — "[TRTLLM-12950][perf] DSv4 follow-up:
  DeepGEMM and MegaMoE", which was not filed against this bug but removed the
  first of its two resident-memory terms (its diff moves `_init_deep_gemm_pdl()`
  out of `torch_custom_ops.py` module scope into `_configure_deep_gemm_pdl()`
  called from `PyTorchModelEngine` init). Two independent unused-memory terms,
  one capacity threshold, so one case. That DeepGEMM term is the subject of its
  own case —
  [Import-time DeepGEMM PDL init creates a CUDA context](../gemm-and-quantization/deepgemm-pdl-import-time-cuda-context.md),
  which covers the five-nvbug group `6390244` / `6402018` / `6418453` /
  `6419078` / `6419139`. This bug was declared a duplicate of that group, but it
  is cross-linked rather than folded: #15632 removed only the first of its two
  terms, and the vision-tower term needed #16250.
- **Failed attempts:** PR #15985 — filed on the sibling bug 6419139 for the
  *same* DeepGEMM import-context term, originally carrying an equivalent
  deferral fix and later rebased to keep only a
  `tests/unittest/others/test_import_side_effects.py` regression guard · closed
  unmerged by its author: "Closing this PR because the same fix was included in
  15632 which has since merged" — so the import-side-effect *guard* never
  landed with it.
- **Symptom:** ~7.5% Total Token Throughput / Inference Time regression on
  `qwen3.5_9b-bench-pytorch-bfloat16-maxbs:512-maxnt:2048-input_output_len:500,2000`
  on L40S, 1.3.0rc19 → rc20 (bug: Inference Time 413361.469 → 444617.063,
  +7.56%; Total Token Throughput 3096.563 → 2878.882, −7.03%; rerun +7.36% /
  −6.86%, a stable regression). Surfaced via the QA release-over-release
  perf comparison. Two tells that this was capacity and not compute: per-step
  GPU time was unchanged, and only the 500/2000 ISL/OSL variant moved — the
  1000/1000, 2000/500 and 128/128 siblings stayed flat.
- **Root cause:** GPU memory held by a component the workload never uses, taken
  out of the KV cache pool, which is sized from *free* GPU memory at startup.
  Since #15249 (dense) and #14599 (MoE), Qwen3.5 checkpoints with architecture
  `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration`
  resolve to the VLM wrapper, which unconditionally instantiated and loaded the
  vision tower (~0.86 GiB for Qwen3.5-9B) even for text-only serving. The pool
  on L40S shrank 22.55 → 21.69 GiB, dropping the number of concurrently
  schedulable 2500-token requests. This workload sits right at a capacity
  threshold — QA history shows 22.55 GiB ↔ ~3.1k tok/s fast regime, ≤22.34 GiB
  ↔ ~2.87k tok/s slow regime — which is why a sub-GiB loss cost ~7.5% on one
  variant and nothing on its siblings. Full ledger from the bug: rc19 22.55 →
  rc20 22.08 (−0.47 GiB, DeepGEMM import-time CUDA context, fixed by #15632) →
  rc21 21.69 (DeepGEMM restored +0.47, vision tower −0.86 from #15249).
- **How introduced:** #15249 "[TRTLLM-13383][feat] Add support for Qwen3.5 VL
  Dense" (and #14599 for MoE) — a new VLM feature changed which model class the
  *existing* text-only checkpoint resolves to, so text-only benchmarks
  inherited the encoder's footprint. The DeepGEMM term came from #15402, which
  added an import-time `deep_gemm.set_pdl()` call (per PR #15985).
- **Fix mechanism:** new prototype `TorchLlmArgs.disable_mm_encoder` skips
  instantiating and loading a VLM checkpoint's multimodal encoder (mirroring
  the existing MM E/P disagg encoder-skip path, gated in the shared
  `Qwen3VLModelBase.__init__`; raw image/video requests are rejected by the
  pre-existing guard). `trtllm-bench throughput` sets it automatically for the
  PyTorch backend when `--modality` is not given — overridable via
  `extra_llm_api_options` — restoring pre-#15249 behavior without QA test
  changes. KV-capacity profiling (`_create_dummy_encoder_inputs`) now skips the
  dummy encoder pass when `model.mm_encoder is None`, which would otherwise
  crash. Verified on H100 (Qwen3.5-9B bf16, maxbs 512 / maxnt 2048 /
  max_seq_len 2500, identical binaries, Python-only pinned-binary bisect): KV
  pool 54.04 GiB / 1,374,848 tok pre-#15249, 53.13 GiB / 1,344,960 tok with the
  tower loaded, 54.03 GiB / 1,374,432 tok with the fix (torch weights-load
  16.94 → 17.87 → 16.97 GiB).
- **Detection signal:** the reported KV pool / block count drops across builds
  while the *config* is unchanged and per-step GPU time is flat — the memory is
  genuinely gone, not mis-estimated, so look for a new resident allocation
  before KV sizing rather than an estimator bug. Check what the checkpoint
  resolves to and whether an encoder is being built:
  `grep architectures <model_dir>/config.json` (a
  `*ForConditionalGeneration` arch on a workload you believe is text-only is
  the tell), then `grep -n "disable_mm_encoder" tensorrt_llm/bench/benchmark/throughput.py tensorrt_llm/_torch/models/modeling_qwen3vl.py`
  and confirm the run logs `multimodal encoder disabled
  (disable_mm_encoder=True); serving text-only requests.` For the import-side
  term, `python -c "import tensorrt_llm"` must create no CUDA context. A
  sub-GiB delta only matters near a threshold, so plot the metric against
  recorded `kv_cache_size` across builds instead of judging the GiB loss alone.
- **Prevention/guard:** PR #16250 added
  `tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl.py::test_qwen35_dense_vl_disable_mm_encoder_skips_vision_tower`
  and `::test_qwen35_dense_vl_default_keeps_vision_tower` (asserting
  `model.mm_encoder is None` and no `mm_encoder` parameters, plus the
  default-on case), a mutual-exclusion validator against `mm_encoder_only`, and
  an api_stability reference entry. Gaps: the flag is **opt-in per model** —
  only the Qwen3-VL / Qwen3.5-VL family honors it and any other model silently
  ignores it — there is no bar asserting that a text-only workload's KV pool is
  unchanged when its checkpoint gains an encoder, and the import-side-effect
  guard from PR #15985 was closed unmerged, so nothing prevents a new
  import-time device allocation from re-shrinking the pool.
- **Generalizes to:** `pattern-unaccounted-startup-residency` — memory a
  workload never uses is resident when the KV pool is sized from free memory,
  so capacity (not kernel speed) regresses. Carries to: any checkpoint whose
  architecture string routes it to a *multi-modal* wrapper for text-only
  serving (audio/video towers, cross-attention adapters); import-time or
  init-time device side effects that create CUDA contexts or workspaces in
  every process, including ones that never launch a kernel; draft/eval models
  or LoRA adapters instantiated for a config that never invokes them; and any
  workload sitting near a KV-capacity threshold, where a sub-GiB footprint
  change flips a whole scheduling regime and looks GPU- or model-specific.
