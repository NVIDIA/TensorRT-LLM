<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Engine features in Modeling V2

A target serves the engine features its deployment turns on, such as the decode
CUDA graph, chunked prefill, speculative decoding or LoRA. `_features.py`
records which step targets serve which features, and `modeling_v2_resolve`
checks the record once `route` has returned a target. This page describes the
three kinds of feature, lists the features each kind holds and is planned to
hold, and records what every routed target does with them today.

## Feature kinds

Features are basic or advanced, by whether every target is expected to serve
them, and advanced features are functional or perf, by what goes wrong when a
target does not serve one:

- **Basic** features are the foundational features every target is expected to
  serve, so a target serves them by default. A target that does not is the
  exception and says so, and turning the feature on for it is an error.
- **Advanced** features are ones some users need but not every target is
  required to serve, so a target serves one only once it declares it. They come
  in two kinds:
  - **Advanced functional** features change what the engine computes. A target
    that ignores one returns wrong results, so only a target that declares it is
    built; otherwise `auto` falls back to Modeling V1.
  - **Advanced perf** features change only how fast the engine runs. A target
    that does not declare one still computes the right result in principle, so
    it is built and the gap is reported.

| Kind | A step target serves it | Declared with | When the route's target has a step target that does not serve it |
|---|---|---|---|
| `BasicFeature` | unless it declares otherwise | `@Feature.unsupported(when=...)` | `ValueError` in every mode |
| `AdvancedFunctionalFeature` | only once it declares it | `@Feature.supported(when=...)` | the target is refused: `auto` builds Modeling V1, `require` raises |
| `AdvancedPerfFeature` | only once it declares it | `@Feature.supported(when=...)` | the target is built, with a warning |

A declaration describes the step target. It does not replace a check in
`route`: when a feature setting makes a target fail (rather than run slower),
`route` (or a `check_deployment` it calls) still raises `NoTarget` for it. The
qwen4_exp target, for example, both declares the prefill CUDA graph settings it
serves and refuses the others in `check_deployment`; the warning then fires only
if a route misses one.

`from_args` sees only the engine arguments, so a feature that a request turns on
(multimodal input, per-request context logits) is checked through the engine
argument that enables it, or not at all.

## Feature list

Status: **in code** features exist in `_features.py`; **planned** ones are
proposed and not implemented yet; **to verify** ones need a look at the targets
before they are classified.

### Basic features

| Feature | Turned on by | Status | Targets that do not serve it today |
|---|---|---|---|
| `DecodeCudaGraph` | `cuda_graph_config` | in code | none. Serving it includes running the dummy rows that `cuda_graph_config.enable_padding` adds to a batch; a `PrefillTarget` serves it without doing anything, since the graph captures only steps without context rows. |
| `ChunkedPrefill` | `enable_chunked_prefill` | in code | none |
| `OverlapScheduler` | `disable_overlap_scheduler=False` | in code | none (GLM-5.2 sm103 tp8 fast paths fall back without it) |
| `KvCacheBlockReuse` | `kv_cache_config.enable_block_reuse` | in code | Kimi K3 tp16 (both): asserted off at the first forward; should declare `unsupported` |

### Advanced functional features

| Feature | Turned on by | Status | Today |
|---|---|---|---|
| `SpeculativeDecoding` | `speculative_config` | planned | Every target serves one mode, checked in `route`: one-model MTP-Eagle (DeepSeek R1, Qwen3.8-Flash-Next, the four GLM-5.2 targets, GLM-5.3-Flash mtp_fused), one-model Eagle3 (MiniMax-M3), SA/DFlash/DSpark (Kimi K3 moetp4ep4), MTP or SA (GLM-5.3-Flash default), none (GPT-OSS, Kimi K3 moetp16ep1). Dynamic trees, `draft_len_schedule` and two-model drafting are refused everywhere; GLM-5.2 sm103 tp8 requires MTP. The feature would carry the `SpeculativeDecodingMode` (`speculative_config.spec_dec_mode`), linearity and draft length so targets state these limits with `when`. |
| `LoRA` | `enable_lora` / `lora_config` | planned | No target serves it. Qwen3.8-Flash-Next asserts and MiniMax-M3 raises at runtime; DeepSeek R1, GPT-OSS and GLM-5.2 sm103 tep4 ignore `lora_params` without a check. |
| `BeamSearch` | `max_beam_width > 1` | planned | No target serves it. Qwen3.8-Flash-Next asserts width 1; DeepSeek R1 and GLM-5.2 sm103 tep4 assume width 1 without a check. |
| `ReturnRoutedExperts` | `enable_return_routed_experts` | to verify | |
| `GuidedDecoding` | `guided_decoding_backend` | planned | GLM-5.3-Flash mtp_fused refuses it in `route`. |
| `ReturnGenerationLogits` | `gather_generation_logits` | planned | |

### Advanced perf features

| Feature | Turned on by | Status | Today |
|---|---|---|---|
| `PrefillCudaGraph` | `prefill_cuda_graph_backend` | in code | Declared by GLM-5.2 sm103 tp8 and sm107 low_latency (all backends), Kimi K3 tp16 (breakable), MiniMax-M3 sm107 tp2 (breakable; decode left uncompiled), Qwen3.8-Flash-Next (breakable under attention TP; decode left uncompiled). MiniMax-M3 sm103 tp4/tp8 and sm107 tp4/tp8 raise at runtime under any prefill CUDA graph and need a refusal in `route`. |
| `MoeLoadBalancer` | `moe_config.load_balancer` | planned | Qwen3.8-Flash-Next (with expert parallelism) and GLM-5.2 sm107 high_throughput refuse it because it would do nothing; a warning fits. |
| `CudaGraphSamplingVariants`, `InGraphSampling` | `cuda_graph_config.sampling_variants`, `enable_in_graph_sampling` | to verify | |
| `AttentionDpBalance` | `attention_dp_config` | to verify | |

### Kept in `route`

These are not features: a target is named after them, or they are part of the
numerics a checkpoint is served with. `route` keeps checking them.

- Parallel layout: pipeline parallelism, context parallelism and helix,
  attention DP, `enable_lm_head_tp_in_adp`, MoE TP×EP.
- Numerics: KV cache dtype, `mamba_ssm_cache_dtype`, model dtype, quantization
  format, `sparse_attention_config`, `kv_cache_compression_config`.

The engine settings the model never sees (scheduler policy, `batch_wait_*`,
`stream_interval`, statistics, tokenizer, the sampler's
`batched_logits_processor` and `watermark_config`, and the KV cache transfers
behind `cache_transceiver_config` and `kv_connector_config`) are neither
features nor checked in `route`. A target that fuses sampling into its draft loop bypasses
the sampler; `SpeculativeDecoding` states that limit with `when`.

Nor are the kernel choices a target makes for itself, such as
`allreduce_strategy`, `moe_config.tactic`, `moe_config.backend`,
`use_cute_dsl_blockscaling_mm/bmm`, `enable_min_latency` and
`torch_compile_config.enable_userbuffers`: they are Modeling V1 settings, and a
Modeling V2 target picks its own kernels.

## Known gaps

Found while recording the targets above; each is a change of its own.

- **Hard errors that `auto` cannot fall back from.** These are assertions or
  `ValueError`s at construction, not `NoTarget` in `route`: Kimi K3 moetp16ep1
  with any speculative decoding; Kimi K3 moetp4ep4 with MTP, Eagle3 or NGram;
  Kimi K3 with a KV cache other than bf16; MiniMax-M3 sm107 with a sparse
  attention or Eagle3 configuration it does not serve.
- **Runtime failures under a perf feature.** MiniMax-M3 sm103 tp4/tp8 and sm107
  tp4/tp8 under any prefill CUDA graph.
- **Silent wrong results.** LoRA and beam search on the targets listed above;
  the planned `LoRA` and `BeamSearch` features close them. GLM-5.3-Flash
  mtp_fused drops a request's `return_context_logits`; a request sets it, so no
  feature sees it, and the target has to refuse the request itself.
