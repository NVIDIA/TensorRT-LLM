---
id: case-qwen35-ssm-cache-fp32-fallback
type: regression-case
family: kernel-and-fusion
module: kv-cache-manager
maturity: full
regression_class: [fast-path-fallback]
signals: [throughput-drop, perf-ci-bar-failure, slower-kernel-in-trace]
subsystems: [kv-cache, runtime-python]
introduced_via: [new-feature]
phase: [decode]
patterns: [pattern-fast-path-silent-fallback]
nvbugs: ["6422332"]
commits: ["d163e74407cd"]
success_prs: [16065]
failed_prs: []
---

# Qwen3.5 GDN state cache allocated in fp32, disabling the bf16-state decode kernel

> Part of the [KV-cache manager regression cookbook](index.md) · schema: [case-template](../case-template.md)

- **Provenance:** nvbug `6422332` · commit `d163e74407cd` · PR #16065 —
  "Keep SSM cache in weights dtype when `mamba_ssm_cache_dtype` is auto".
- **Symptom:** Qwen3.5 serving throughput down — the fix diff's docstring
  states "~20% serving throughput loss" from the fp32 state cache. The PR
  description states no further number.
- **Root cause:** `resolve_mamba_ssm_cache_dtype` in
  `tensorrt_llm/_torch/pyexecutor/config_utils.py` accepted **either**
  `mamba_ssm_cache_dtype` **or** `mamba_ssm_dtype`, on the top-level config or
  the nested `text_config`, as the dtype for hybrid SSM-state cache allocation.
  The Qwen3.5 checkpoint declares `mamba_ssm_dtype=float32`, which expresses the
  SSM *compute* dtype intent, not a cache-storage request — so with
  `mamba_ssm_cache_dtype="auto"` the GDN linear-attention state cache was
  allocated in fp32 instead of the bf16 weights dtype. Per the docstring the fix
  adds, that "disables the FlashInfer bf16-state decode kernel and doubles state
  memory traffic".
- **How introduced:** PR #14599 · commit `21260bbc0b34`
  ("[TRTLLM-12500][feat] Add support for Qwen3.5 VL MoE (with the MTP fixes)"),
  per git history — it added `resolve_mamba_ssm_cache_dtype`. Before it, the
  `"auto"` branch of `validate_and_set_mamba_ssm_cache_dtype`
  (`pyexecutor/model_loader.py`) read only the top-level `mamba_ssm_cache_dtype`
  and otherwise fell back to `pretrained_config.torch_dtype`; #14599 centralized
  the lookup into the new `resolve_mamba_ssm_cache_dtype`, which additionally
  consults `mamba_ssm_dtype` and `text_config` — so a checkpoint field that had
  never reached this path started driving cache allocation.
- **Fix mechanism:** renames the helper to `resolve_ssm_cache_dtype` and narrows
  it to the explicit `mamba_ssm_cache_dtype` field only (still checking
  top-level and `text_config`). With `"auto"`, resolution now falls through to
  `resolve_hf_torch_dtype` / the model's weights dtype, keeping the state cache
  in bf16; fp32 states remain available as a deliberate opt-in via
  `kv_cache_config.mamba_ssm_cache_dtype`. Both call sites
  (`extract_mamba_kv_cache_params`, `validate_and_set_mamba_ssm_cache_dtype`)
  are repointed, and the reasoning is written into the helper docstring.
- **Detection signal:** on a recurrent-state hybrid (Mamba SSM, Qwen3.5 GDN
  linear attention) the resolved state-cache dtype is fp32 while the weights are
  bf16, and decode runs the generic rather than the bf16-state kernel. Inspect
  the resolution chain with
  `grep -n "mamba_ssm_cache_dtype\|mamba_ssm_dtype" tensorrt_llm/_torch/pyexecutor/config_utils.py tensorrt_llm/_torch/pyexecutor/model_loader.py`,
  and assert the `"auto"` outcome with
  `pytest tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py -k resolves_mamba_ssm_cache_dtype`.
- **Prevention/guard:** PR #16065 updated
  `test_qwen35_moe_vl_resolves_mamba_ssm_cache_dtype`
  (`tests/unittest/_torch/modeling/test_modeling_qwen3_5_vl_moe.py`) to assert
  bf16 under `"auto"` and added an explicit-`"float32"` opt-in assertion, and
  flipped the same assertion in
  `tests/unittest/_torch/modeling/test_qwen_image_bench_modeling.py`; both
  carry a comment naming the perf consequence. Gap: nothing logs at runtime when
  a checkpoint's declared SSM dtype differs from the cache dtype — review asked
  for such a warning and the merged diff has none — so the same silent fp32
  promotion arriving through another config field would again surface only as a
  tripped perf bar.
- **Generalizes to:** `pattern-fast-path-silent-fallback` — widening a dtype by
  one notch makes a dtype-specialized kernel ineligible; the run stays correct,
  just slower. Carries to: compute-dtype vs storage-dtype conflation for any
  cache (KV, Mamba conv state, spec-decode hidden state); config-lookup helpers
  "centralized" during new-model enablement that begin honoring extra field
  aliases or nested `text_config` values; `"auto"` dtype-resolution chains whose
  fallback order decides which kernel is eligible; checkpoint-declared fields
  treated as runtime policy with no explicit opt-in gate.
