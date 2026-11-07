---
id: case-fp8-mla-kv-cache
type: case
family: kernel-and-fusion
maturity: full
bottleneck: [memory, compute]
signals: [kv-reads-dominate, long-context, high-concurrency, memory-capacity-bound]
architectures: [sm90, sm100]
model_scope: [mla, deepseek-v3r1, model-agnostic]
phase: [decode]
patterns: [pattern-quantize-memory-bound-side]
accuracy_risk: lossy
apply_via_kind: [config-knob]
knobs: [kv_cache_config.dtype]
specialists: [kernel-cuda-specialist, perf-nsight-compute-analysis]
commits: ['515dd0d78fe8', '897c4bffd7']
eligibility:
  - "sm == 90 or sm == 100 (mFP8GenerationMLA gate: mSM==90||100)"
  - "MLA model with FP8 KV-cache mode (QuantMode.has_fp8_kv_cache()); generation path only — context (prefill) MLA stays BF16"
interactions:
  - {feature: fp8-context-fmha, relation: incompatible-with, note: mFP8GenerationMLA and mFP8ContextFMHA are mutually exclusive}
measured: []
---

# Quantize MLA generation to FP8 KV cache while keeping BF16 output

> Part of the [Kernel & Fusion casebook](index.md) · schema: [case-template](../case-template.md)

- **Commits:** `515dd0d78fe8` feat: Add support for FP8 MLA on Hopper and Blackwell (#3190); related: `897c4bffd7` Integrate FP4 indexer for DSA on Blackwell (#13340, the same idea applied to a *selector* cache one precision step lower).
- **Applies when:** memory-bound MLA decode — KV-cache reads dominate generation; DeepSeek-style MLA (kv_lora_rank + qk_rope_head_dim latent KV, head_size 576); SM 90/100; long context / high concurrency where KV footprint and HBM bandwidth limit decode. Context (prefill) MLA stays BF16; the win is the generation path.
- **Mechanism:** stores latent KV as `__nv_fp8_e4m3` (half the KV bytes + per-step bandwidth), runs the generation FMHA/FlashMLA with `dataTypeKv=DATA_TYPE_E4M3`. Q is quantized into `quant_q_buffer` with two BMM scales (`mla_bmm1_scale`/`mla_bmm2_scale`); **output is written BF16** (`dataTypeOut=DATA_TYPE_BF16`) so o_proj input stays high-precision and loss is confined to the KV/Q read path. `mFP8GenerationMLA` and `mFP8ContextFMHA` mutually exclusive; gated to `mSM==90||100`.
- **Generalizes to:** "quantize the memory-bound side of attention (KV cache + matching operand) to FP8 but keep accumulation/output in BF16"; carries to GQA/MHA FP8 KV, FP8 context FMHA, FP4/INT8 KV; adapt by choosing the kv dtype the FMHA kernel supports, supplying BMM/dequant scales, and keeping output dtype unquantized where the next op (o_proj) is precision-sensitive. Also carries to a **model-specific *selector* cache**: DSA's indexer K cache goes one step lower to **FP4 (packed E2M1 + block-32 UE8M0)**, halving its per-token footprint (~132→68 B) and pairing a `fused_cat_fp4` cache-write with FP4×FP4 logits kernels (`indexer_k_dtype="fp4"`, SM≥100; #13340) — a selector cache tolerates coarser precision than the value KV because its output only ranks (cf. [ranking-only precision](ranking-only-precision-tf32.md)).
- **Apply via:** `KvCacheConfig.dtype="fp8"` (`QuantMode.has_fp8_kv_cache()` → `mFP8GenerationMLA`); no separate gen-MLA knob (derived from FP8 KV mode on an MLA model). Delegate kernel work to **kernel-cuda-specialist**; profile with **perf-nsight-compute-analysis**.
- **Expected effect:** lower KV HBM bytes, higher MLA decode throughput / lower latency under bandwidth pressure; no number — measured Δ to be recorded from run.
- **Accuracy risk:** lossy (FP8 e4m3 KV + FP8 Q); output BF16 but KV/Q rounding perturbs logits — needs accuracy record (GSM8K/MMLU vs BF16-KV) + rollback criterion before promotion.
- **Verify:** decode throughput + KV bytes/step; accuracy parity vs `kv_cache_config.dtype="auto"`; confirm output stays BF16 (code asserts FP8-KV MLA requires bf16 output).
- **Rollback:** `kv_cache_config.dtype="auto"`. Trigger: accuracy regression beyond recorded threshold, or FP8 gen-MLA kernel unavailable for the SM/head config.
- **Prior art:** PR #3190. Files: `cpp/.../common/attentionOp.cpp` (`mFP8GenerationMLA`), `kernels/mlaKernels.cu`, `kernels/flashMLA/flash_fwd_mla_fp8_sm90.cu`, `_torch/attention/mla.py`. Owning specialist: **kernel-cuda-specialist**.
