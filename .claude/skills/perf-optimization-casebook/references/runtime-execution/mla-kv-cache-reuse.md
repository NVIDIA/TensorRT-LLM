---
id: case-mla-kv-cache-reuse
type: case
family: runtime-execution
maturity: full
bottleneck: [compute, memory]
signals: [attention-hot-path, long-context]
architectures: [sm90, sm100]
model_scope: [model-agnostic, mla, deepseek-v3r1]
phase: [prefill]
patterns: [pattern-reuse-computed-state]
accuracy_risk: mixed
apply_via_kind: [config-knob, default-on]
knobs: [enable_block_reuse]
specialists: [trtllm-serve-config-guide, perf-sweep-workflow]
commits: ['97bc680cd8a5', '8452775db86b', 'a891013e3c75']
eligibility:
  - "SM90–SM100 only; KV-cache quant must be none or FP8 (py_executor_creator.py gate)"
measured: []
---

# Reuse cached/prefix KV for MLA to skip down-projection recompute (MLA KV-cache block reuse)

> Part of the [Runtime / Execution casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `97bc680cd8a5` feat: support kv cache reuse for MLA (#3571); related: `8452775db86b` Support FP8 KV Cache Reuse for MLA (#4535), `a891013e3c75` Optimize KV Cache Reuse for MLA (#4869).
- **Applies when:** compute-bound prefill with repeated prefixes across requests, MLA attention, SM90/SM100. Signals: high cache-hit potential (shared prompts, chat history), large ISL relative to OSL. Enabled by default; originally excluded FP8 KV cache (#4535 lifted that); plain non-FP8 quantized KV still disables reuse.
- **Mechanism:** MLA stores the *compressed* latent KV (`compressed_kv` + `k_pe`) in the paged KV cache. On a prefix/block hit, cached tokens skip the recompute and run only the up-projection GEMM, using a 192/128 head-size MLA context kernel; shared with chunked context (`mla_context_paged_kv`, `prepare_paged_context_mla`). #4869 fuses the write path into one `invokeMLARopeAppendPagedKVAssignQ` kernel (RoPE + append-paged-KV + assign-Q), drops redundant `.contiguous()` copies, collapses to one up-projection GEMM — cutting the memcpy/GEMM overhead the feature otherwise adds.
- **Generalizes to:** "cache and reuse the compressed/intermediate representation across requests to skip recompute"; carries to standard MHA/GQA block reuse, chunked-context prefill reuse, prefix caching for any attention variant; adapt by choosing what to persist (full KV vs MLA's compressed latent) and gating on hardware/kernel availability.
- **Apply via:** `KvCacheConfig(enable_block_reuse=True)` (default on for MLA on SM90/SM100). Auto-disabled outside SM90–SM100 and for KV-cache quant other than none/FP8 (`py_executor_creator.py` gate). Delegate the YAML knob to **trtllm-serve-config-guide**; measure on a prefix-heavy trace with **perf-sweep-workflow**.
- **Expected effect:** higher prefill throughput / lower TTFT on prefix-overlapping traffic. README caveat: "GPU memory consumption may be higher and the E2E performance may have regression in some cases" (extra memcpy + GEMMs) — workload-dependent; measured Δ to be recorded from run.
- **Accuracy risk:** lossless for bf16/non-quant reuse; with FP8 KV reuse (#4535) the stored latent is FP8 — lossy in the same way FP8 KV cache already is, not an additional reuse-specific loss.
- **Verify:** prefill throughput / TTFT + KV-cache-hit rate on a repeated-prefix trace; GPU memory headroom. For FP8 KV reuse run an accuracy/parity check vs bf16 KV.
- **Rollback:** `KvCacheConfig(enable_block_reuse=False)`. Trigger: E2E regression on a low-reuse workload, memory pressure shrinking batch/KV, or accuracy drift with FP8 KV.
- **Prior art:** PRs #3571, #4535, #4869. Files: `cpp/.../kernels/mlaKernels.cu` (`invokeMLARopeAppendPagedKVAssignQ`), `thop/mlaPreprocessOp.cpp`, `_torch/attention_backend/trtllm.py` (`prepare_paged_context_mla`), `pyexecutor/py_executor_creator.py` (SM/quant gating).
