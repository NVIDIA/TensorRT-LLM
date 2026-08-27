<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-Flash-Next BF16 and Block-FP8 Feature Support

## Overview

This document describes the TensorRT-LLM PyTorch-backend feature support for
Qwen3.8-Flash-Next BF16 and pre-quantized block-FP8 checkpoints. The Hugging Face
conditional-generation architecture is `Qwen4ExpForConditionalGeneration`;
language-only serving resolves the decoder as `Qwen4ExpForCausalLM`.

The model combines QSA sparse full-attention layers, Gated DeltaNet recurrent
layers, Hyper-Connections, PLE recurrent state, and routed and shared experts.
These components have state and parallelism requirements beyond those of a
conventional decoder-only transformer and require model-specific state ownership
throughout prefill, decode, cache reuse, and disaggregated serving.

This support statement covers BF16 and pre-quantized, 128-by-128 block-scaled
FP8 routed-expert weights. NVFP4 is not included.

### Official checkpoints

The official Hugging Face checkpoints are:

- BF16: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Block-FP8: [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)

## How to read the support claims

A server reaching the ready state is not sufficient validation for this model.
A **Validated** claim requires semantic output checks and evidence that the
intended model-specific runtime path executed.

| Status | Meaning |
|---|---|
| **Validated** | Exercised with a production-size checkpoint on NVIDIA GB300 GPUs, including semantic output and runtime-path checks. |
| **Validated with constraints** | Validated only for the topology or configuration constraints stated in the corresponding matrix note. |
| **Supported** | Implemented and covered by focused tests or by the same validated runtime contract, but the exact precision/topology combination has not received an independent production-size run. |
| **Implemented; validation pending** | The model integration and configuration path exist, but a production-size semantic end-to-end run has not passed the release gate. This is not a release support claim. |
| **Not validated** | No release claim is made for this precision/topology combination. |
| **Not applicable** | The feature does not match this model architecture or checkpoint format. |
| **Out of scope** | Deliberately excluded from this support package. |

## Feature support matrix

### Model architecture and inference paths

| Feature | BF16 checkpoint | Block-FP8 checkpoint | Notes |
|---|---|---|---|
| Text generation | **Validated** | **Validated** | OpenAI-compatible serving, deterministic endpoint checks, long-context retrieval, and accuracy workloads have been exercised. |
| Image-language inference | **Validated with constraints** | **Validated with constraints** | Aggregate serving with a local encoder is validated for BF16 TP4 and block-FP8 TP1. Both precisions passed single-image, ordered multi-image, sequential, and concurrent semantic checks. A separate multimodal encoder-to-prefill handoff is not claimed. |
| QSA sparse attention | **Validated** | **Validated** | Both exact and fused paged sparse paths are exercised above the configured sparse threshold. Dense full attention is not used as substitute acceptance evidence. |
| Gated DeltaNet | **Validated** | **Validated** | Prefill, decode, chunk continuation, cache compaction, request reuse, and speculative-state promotion are covered. |
| Hyper-Connections | **Validated** | **Validated** | Multi-stream residual mixing and topology-specific reductions are part of the accepted end-to-end paths. |
| PLE | **Validated** | **Validated** | Token and short-convolution state are managed per request and participate in reuse, offload, MTP promotion, and text disaggregation where applicable. |
| PLE embedding-weight sharding and pinned-host offload | **Validated** | **Validated** | The 320,001,536-row table can remain row-sharded in device memory or move to pinned host memory. The host path gathers only the 16 selected rows per token and overlaps the UVA gather with the first decoder block. |
| QSA/GDN/PLE state with KV cache manager V2 | **Validated** | **Validated** | KV cache manager V2 is the required lifecycle owner for the model-specific auxiliary state. |

### Parallelism, communication, and speculative decoding

| Feature | BF16 checkpoint | Block-FP8 checkpoint | Notes |
|---|---|---|---|
| Tensor parallelism | **Validated** | **Validated with constraints** | BF16 TP4 is validated. Block-FP8 TP1 is validated on one GB300. For block-FP8, routed-expert tensor-parallel partitions must preserve 128-element scale blocks. |
| Tensor/expert parallelism (TEP) | **Validated** | **Validated** | BF16 TEP2 and TEP4 are validated. Block-FP8 also has validated two- and four-GPU TEP topologies. Use `moe_tensor_parallel_size: 1` and expert parallelism when a pure MoE-TP split would cut a scale block. |
| Attention data parallelism (ADP) | **Validated** | **Validated** | BF16 ADP2/EP2 and ADP4 are validated with expert-parallel routed experts and model-specific recurrent-state ownership. Small topology-sensitive BF16 numerical differences are expected and are not treated as request-integrity failures. |
| Pipeline parallelism | **Validated** | **Not validated** | BF16 PP4 and PP2+TP2 functional paths are validated. No block-FP8 PP release claim is made by this matrix. |
| MTP with draft depth 3 | **Validated** | **Validated** | The recurrent MTP layer supports a configured maximum draft length of three, including accepted-prefix promotion of GDN, QSA, and PLE state. Greedy decoding uses strict acceptance; distribution-correct non-greedy decoding uses rejection sampling with the full advanced sampler. Other draft depths are not part of this release claim. |
| GDN replay under MTP | **Validated** | **Validated** | Replay and accepted-state commit are covered together with semantic output checks. |
| AllReduce `AUTO` on GB300 | **Validated** | **Validated** | Production configurations may omit `allreduce_strategy`. On SM103, automatic selection excludes the unsupported `NCCL_SYMMETRIC` tactic while retaining NCCL, ONESHOT, and TWOSHOT candidates. Explicit NCCL remains useful as a diagnostic control. |

### MoE backends and load balancing

| Feature | BF16 checkpoint | Block-FP8 checkpoint | Notes |
|---|---|---|---|
| CUTLASS MoE | **Validated** | **Not validated** | CUTLASS is the conservative BF16 backend and is used where the TRTLLM BF16 kernel shape is ineligible. This matrix does not make a block-FP8 CUTLASS claim. |
| TRTLLM MoE | **Validated** | **Validated** | BF16 validation uses a topology with an aligned routed-expert partition. Block-FP8 validation includes TEP4, ADP4, and TP1 execution with the FP8 block-scale runner and no backend fallback. |
| DeepGEMM MoE | **Not applicable** | **Validated** | DeepGEMM requires block-scaled FP8 routed experts and SM100 or SM103. A pinned TP1 GB300 smoke resolved all 48 routed-expert layers to `DeepGemmFusedMoE` without backend fallback. |
| CuteDSL MoE | **Not applicable** | **Not applicable** | The supported BF16 and block-FP8 combinations in this document do not use CuteDSL. A request that falls back to CUTLASS is not evidence of CuteDSL support. |
| Static/offline EPLB | **Validated** | **Not validated** | BF16 expert-parallel serving has loaded and executed an explicit 48-layer placement with replicated slots. Placements are topology and workload specific. |
| Dynamic/online EPLB | **Validated** | **Not validated** | BF16 online observation, placement refresh, and expert updates are functionally validated. Production use should retain migration and stable-window observability. |

### Scheduler, cache, and serving features

| Feature | BF16 checkpoint | Block-FP8 checkpoint | Notes |
|---|---|---|---|
| CUDA graph with padding | **Validated** | **Validated** | Decode graph capture, short/long graph pairs, and padded batch tiers are covered with QSA and model-specific state. |
| Chunked prefill | **Validated** | **Validated** | Validation includes prompts that cross the configured chunk budget; merely enabling the option is not considered sufficient. |
| Overlap scheduler | **Validated** | **Validated** | Concurrent requests, abort/recovery, and slot reuse are covered. |
| Combined graph, chunked prefill, overlap, and QSA | **Validated** | **Validated** | This is a functionality statement. It does not attribute an isolated performance gain to any individual feature. |
| FP8 KV cache | **Validated** | **Validated** | FP8 KV cache is independent of model-weight precision. Both BF16 weights + FP8 KV and block-FP8 routed-expert weights + FP8 KV are valid combinations. |
| Prefix caching (prefill cache / block reuse) | **Validated** | **Validated** | BF16 validation includes repeated-prefix hits and output invariance. Block-FP8 validation covers shared-system-prompt and multi-turn agentic requests with server-side reused-block evidence. Hybrid recurrent state requires an explicit snapshot interval. |
| Host offload/onboard | **Validated** | **Supported** | BF16 validation covers pressure, offload, onboard, and replay of standard and model-specific state. |
| Text prefill/decode disaggregated serving | **Validated** | **Validated** | The validated text path transfers KV, QSA index state, Gated DeltaNet state, and PLE side state between context and generation workers. Block-FP8 validation covers an ADP4/EP4 context worker and a separate ADP4/EP4 generation worker with MTP3. |
| Multimodal encoder/prefill disaggregation | **Not validated** | **Not validated** | No separate multimodal encoder-to-prefill handoff path is claimed. This limitation is independent of the aggregate configuration example below. |
| NVFP4 weights | **Out of scope** | **Out of scope** | No NVFP4 configuration or result is part of this support statement. |

## Validation evidence

The results below are the retained release evidence. They report complete
datasets or explicit smoke-test contracts, preserve request failures and
generation-limit responses in the denominator, and combine client-side semantic
checks with server-side runtime evidence where applicable.

For reproducibility, retain the resolved server configuration and final LLM
arguments, individual responses and finish reasons, per-request performance
records, and backend-resolution logs. A configured feature is not considered
verified unless the effective runtime state and request records show that it ran.

### PLE embedding-weight residency and pinned-host offload

The PLE embedding table has shape `[320001536, 160]`: 51.2 billion parameters,
95.37 GiB in BF16 or 47.68 GiB in FP8. This is fixed model-weight storage, not
KV-cache or mutable recurrent state.

By default the table remains in device memory. Ordinary TP assigns a disjoint
row range to each rank and combines partial embedding activations with an
AllReduce. Attention DP first exchanges row IDs, performs the local-shard
lookup, and ReduceScatters the embedding activations back to their token owners.
The mapper loads only the intersection between each checkpoint shard and the
rank-local row range; it does not materialize a full-table destination first.

The optional pinned-host path preserves the same row ownership and collective
semantics. A Triton UVA kernel gathers the 16 sparse rows selected for each
token into a BF16 device buffer. The gather runs on a dedicated CUDA stream and
starts before the decoder loop, allowing it to overlap the first decoder block.
Stable mapped pointers and graph output buffers allow decode CUDA graph capture
and replay without moving the complete table to HBM.

The retained GB300 measurements are:

| Checkpoint and PLE placement | PLE storage per rank | GPU memory after weight load | Profile peak | Semantic/runtime evidence |
|---|---:|---:|---:|---|
| BF16, legacy replicated device table, TP2 | 95.37 GiB device | 215.79 GiB | 221.53 GiB | Historical resident baseline |
| BF16, row-sharded device table, TEP2 | 47.68 GiB device | 168.10 GiB | 174.29 GiB | Concurrent batch 2, long QSA, CUDA graph, MTP3, and GSM8K 8/8 passed |
| BF16, row-sharded device table, ADP2/EP2 | 47.68 GiB device | 172.02 GiB | 178.17 GiB | Real ID-AllGather/activation-ReduceScatter, concurrent batch 2, long QSA, CUDA graph, MTP3, and GSM8K 8/8 passed |
| BF16, pinned host, TP2 | 47.68 GiB host | 120.42 GiB | 126.17 GiB | Long QSA, CUDA graph, MTP3, and GSM8K 8/8 passed |
| BF16, pinned host, TP1 | 95.37 GiB host | 239.18 GiB | 243.49 GiB | Complete BF16 model fits one GB300; long QSA, CUDA graph, MTP3, and GSM8K 8/8 passed |
| Block-FP8, device table, TP1 | 47.68 GiB device | 169.53 GiB | 173.44 GiB | Resident baseline |
| Block-FP8, pinned host, TP1 | 47.68 GiB host | 121.85 GiB | 125.77 GiB | Long QSA, CUDA graph, and GSM8K 8/8 passed |

The matched block-FP8 TP1 comparison isolates a 47.68-GiB HBM reduction, equal
to the complete FP8 table. For BF16, two-GPU row sharding first reduces the
resident table from 95.37 to 47.68 GiB per rank, and host offload removes that
remaining rank-local shard from HBM. The BF16 TP1 result is also a capacity
result: keeping the additional 95.37-GiB table in device memory would exceed a
276.62-GiB GB300, while the host-offloaded model completed at a 243.49-GiB
profile peak.

These are functionality and memory-capacity measurements, not throughput
claims. The semantic smokes retained every individual response and used eight
GSM8K questions, five shots, temperature 0, seed 42, and a 512-token maximum.
Both new resident-sharding runs additionally completed two simultaneous
deterministic requests and a 4,853-token QSA retrieval request.

This branch exposes PLE weight offload as an opt-in environment setting. Set it
before importing PyTorch or starting serving workers:

```bash
export TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1
export PYTORCH_ALLOC_CONF=pinned_use_cuda_host_register:True,pinned_max_round_threshold_mb:128,pinned_max_cached_size_mb:512,pinned_num_register_threads:8

trtllm-llmapi-launch trtllm-serve /path/to/Qwen3.8-Flash-Next \
  --config serving.yaml \
  --served_model_name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 8000 \
  --generation-config trtllm --no-telemetry
```

The bounded pinned-allocation settings avoid table-sized power-of-two rounding
and retaining a freed table-sized block in the host cache. The host must have
enough pinned-memory capacity for every local serving rank. The path currently
supports the standard Hugging Face `AUTO` loader. Attention DP combined with
context parallelism greater than one is rejected because that PLE token/group
collective contract is not implemented.

### Block-FP8 accuracy

The table reports the best validated block-FP8 score retained for each complete
dataset. All results use the same generation protocol: **thinking enabled**,
**temperature 1.0**, **top-p 0.95**, **maximum generation length 65,536 tokens**,
and **seed 42**.

| Dataset | Correct | Accuracy |
|---|---:|---:|
| GSM8K | 1,295 / 1,319 | **98.18%** |
| AIME26 | 28 / 30 | **93.33%** |
| GPQA Diamond | 183 / 198 | **92.42%** |

Each result covers the full dataset and has zero request errors and zero empty
outputs. These are single-run point estimates selected from the validated
configurations. They are functionality and accuracy evidence, not a comparison
between serving or parallelism strategies.

### BF16 MTP3 acceptance

The table reports unforced BF16 MTP3 acceptance on complete semantic datasets.
Acceptance is the exact aggregate ratio
`accepted_draft_tokens / drafted_tokens` across all request records; it is not
the mean of per-request percentages. Every request record proved a maximum
draft length of three, and forced acceptance was not enabled.

All full-dataset runs used thinking mode, temperature 1.0, top-p 0.95, seed 42,
rejection sampling, and `advanced_sampling_mode: full`. The maximum generation
length was 65,536 tokens for GSM8K and 65,535 tokens for GPQA Diamond. Complete
dataset coverage, request errors, empty responses, and responses reaching the
generation limit were retained in the result denominator.

| Dataset | Serving topology | Requests | Accepted / drafted tokens | Acceptance | Accuracy |
|---|---|---:|---:|---:|---:|
| GSM8K | Aggregate TEP4 | 1,319 | 412,482 / 696,480 | **59.22%** | 1,289 / 1,319 (97.73%) |
| GSM8K | Aggregate ADP4 | 1,319 | 428,171 / 701,925 | **61.00%** | 1,292 / 1,319 (97.95%) |
| GPQA Diamond | Aggregate TEP4 | 198 | 1,731,001 / 3,480,723 | **49.73%** | 179 / 198 (90.40%) |
| GPQA Diamond | Aggregate ADP4 | 198 | 1,763,395 / 3,582,222 | **49.23%** | 179 / 198 (90.40%) |

Acceptance is workload- and configuration-dependent. The measured rates mean
that target verification remains a substantial part of these long,
thinking-enabled sampled workloads; they do not by themselves indicate an
output-correctness failure. Results from different datasets or serving
topologies must not be treated as estimates of a single model constant.

### Block-FP8 prefix caching

A TP1 prefix-caching smoke on one GB300 completed all 60 requests with zero
request errors and zero empty outputs. Server metrics recorded cache reuse for
58 requests and 7,120 reused blocks:

- all 19 shared-system-prompt follow-up requests reused 112 blocks each;
- all 19 later agentic first-turn requests reused 128 blocks each;
- all 20 agentic second-turn requests reused 128 blocks each;
- all 60 scenario-specific semantic checks passed.

The validated cache configuration was:

```yaml
kv_cache_config:
  use_kv_cache_manager_v2: true
  enable_block_reuse: true
  enable_partial_reuse: true
  copy_on_partial_reuse: true
  mamba_state_config:
    periodic_snapshot_interval: 256
```

Without the recurrent-state snapshot policy, the runtime safely disables block
reuse because attention KV blocks alone cannot restore Gated DeltaNet and PLE
state. A submitted setting is therefore not sufficient evidence; verify the
effective LLM arguments and per-request `num_reused_blocks` records.

### Aggregate multimodal acceptance

BF16 TP4 and block-FP8 TP1 aggregate serving were validated with the same
bounded image-language contract. Each checkpoint processed four single-image
cases and one ordered two-image case, first sequentially and then concurrently,
for 10 requests per precision. Every request returned HTTP 200, non-empty
schema-valid JSON, the expected semantic result, and a normal stop reason.

The block-FP8 TP1 run used one GB300 backend GPU, the local multimodal encoder,
TRTLLM MoE, QSA, and KV cache manager V2. Runtime evidence confirmed the
pre-quantized 128-by-128 FP8 scale-block metadata, FP8 block-scale MoE execution,
QSA exact and fused sparse paths, `is_disagg=False`, PLE state bound to the V2
lifecycle, and a disabled cache transceiver. Sequential and concurrent requests
produced the same parsed answers; JSON whitespace is not part of the semantic
contract. This result validates aggregate image-language inference only, not a
separate encoder-to-prefill handoff.

### Runtime-path checks

| Check | Observed evidence | Release conclusion |
|---|---|---|
| Block-FP8 TP1 capacity | Approximately 221 GiB model-profile peak on a GB300 with approximately 277 GiB usable memory; approximately 51 GiB remained for cache and runtime allocations. | The complete checkpoint fits functionally on one GB300. This is not a concurrency recommendation. |
| DeepGEMM MoE | All 48 routed-expert layers resolved to `DeepGemmFusedMoE` with no fallback; endpoint, concurrency, repeat-output, and 8/8 GSM8K semantic checks passed. | The block-FP8 TP1 DeepGEMM path is validated on SM103. |
| QSA | Exact and fused paged sparse execution ran above the sparse threshold. | Dense full attention was not accepted as substitute evidence. |
| MTP state lifecycle | Draft depth 3, GDN replay, and accepted-prefix promotion of QSA, GDN, and PLE state were observed with semantic output checks. | MTP3 state promotion is validated; other draft depths are not claimed. |
| Text disaggregation | NIXL setup and transfer of KV, QSA index state, Gated DeltaNet state, and PLE side state were observed across context and generation workers. | Text prefill/decode disaggregation is validated. |
| Combined serving features | CUDA graph padding, chunked prefill, overlap scheduling, QSA, and FP8 KV cache ran together with semantic checks. | Feature coexistence is validated; no isolated performance gain is claimed. |

## Deployment constraints

### Block-FP8 checkpoint format

The block-FP8 checkpoint is pre-quantized. It is not produced by setting an FP8
option while loading the BF16 checkpoint. Routed-expert projections carry FP8
weights and inverse scales with 128-by-128 blocks, while modules excluded by the
checkpoint quantization metadata remain in their original precision. The runtime
must preserve that metadata during weight mapping and must select an MoE backend
that consumes `FP8_BLOCK_SCALES`.

### Block-aligned expert sharding

The 640-element routed-expert intermediate dimension is five 128-element blocks.
This makes MoE-TP1 block aligned. A pure MoE-TP2 split produces 320 elements per
rank, and a pure MoE-TP4 split produces 160 elements per rank; neither is a
multiple of 128. Those partitions are rejected before weight loading because a
naive weight/scale shard would split quantization blocks. Valid multi-GPU choices
include expert parallelism with `moe_tensor_parallel_size: 1`, such as TEP2 or
TEP4.

### One-GPU memory envelope

A complete block-FP8 model has been validated on one GB300. The observed model
profiling peak was approximately 221 GiB on a device with approximately 277 GiB
of usable memory, leaving approximately 51 GiB for cache and runtime allocations
under the validated configuration. This establishes functional fit, not a
high-concurrency performance recommendation.

### Weight precision and KV-cache precision

FP8 KV cache is a separate setting. The following are distinct supported cases:

- BF16 model weights with FP8 KV cache;
- pre-quantized block-FP8 routed-expert weights with an automatic/BF16 KV cache;
- pre-quantized block-FP8 routed-expert weights with FP8 KV cache.

## Example deployment recipes

The examples below are functionality-oriented, end-to-end starting points. They
cover the primary requested deployment shapes: BF16 TEP2 with MTP3, block-FP8
TP1 with MTP3, text prefill/decode disaggregation, and aggregate multimodal
serving. Capacity fields must be sized for the selected checkpoint, GPU count,
input/output lengths, and concurrency. Performance-sensitive deployments should
tune from a matched validated baseline.

### Choosing a recipe

| Deployment goal | Starting point | Validation status |
|---|---|---|
| BF16 on two GB300 GPUs with MTP3 | BF16 TEP2 with MTP3 | Validated with a resident row-sharded PLE table at batch size 2; resize capacity fields for larger workloads |
| BF16 attention-DP on two GB300 GPUs with MTP3 | BF16 ADP2/EP2 with MTP3 | Validated with a resident row-sharded PLE table and real ID-AllGather/activation-ReduceScatter |
| Block-FP8 on one GB300 with MTP3 | Block-FP8 TP1 with MTP3 | Validated |
| BF16 tensor-parallel text serving | BF16 TP4 with CUTLASS MoE | Validated |
| BF16 expert-parallel text serving | BF16 TEP4 with TRTLLM MoE | Validated |
| Block-FP8 multi-GPU text serving | Block-FP8 TEP4 with TRTLLM MoE | Validated |
| MTP3 | Add the greedy or non-greedy MTP3 overlay | Validated |
| Prefix caching | Use the complete cache configuration in the Validation evidence section | Validated |
| CUDA graph, chunked prefill, overlap, and FP8 KV cache | Add the combined feature overlay | Validated for coexistence |
| BF16 text disaggregation | BF16 TP4 context and generation workers | Validated |
| Block-FP8 text disaggregation | Block-FP8 expert-parallel context and generation workers | Validated |
| BF16 image-language serving | Multimodal aggregate example | Validated with a local encoder |
| Block-FP8 image-language serving | Multimodal aggregate example with the block-FP8 TP1 overlay | Validated on one GB300 backend GPU |

### Aggregate text generation

#### BF16 TEP2 with MTP3

This two-GPU topology uses attention TP2 and routed-expert EP2. Keeping
`moe_tensor_parallel_size` at one preserves each expert's complete intermediate
dimension, while `moe_expert_parallel_size: 2` partitions the routed experts
between the two GPUs. Save the following configuration as
`bf16-tep2-mtp3.yaml`:

```yaml
tensor_parallel_size: 2
moe_tensor_parallel_size: 1
moe_expert_parallel_size: 2
pipeline_parallel_size: 1
enable_attention_dp: false
disable_mm_encoder: true

max_batch_size: 8
max_input_len: 8192
max_seq_len: 16384
max_num_tokens: 8192
enable_chunked_prefill: true
disable_overlap_scheduler: false

cuda_graph_config:
  enable_padding: true
  max_batch_size: 8

moe_config:
  backend: TRTLLM
  max_num_tokens: 8192
  disable_finalize_fusion: true

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  max_tokens: 65536
  avg_seq_len: 8192
  use_kv_cache_manager_v2: true
  enable_block_reuse: false

speculative_config:
  decoding_type: MTP
  max_draft_len: 3
  use_rejection_sampling: true
  advanced_sampling_mode: full
```

Start the two-rank aggregate server with:

```bash
trtllm-llmapi-launch trtllm-serve /path/to/Qwen3.8-Flash-Next \
  --config bf16-tep2-mtp3.yaml \
  --served_model_name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 8000 \
  --generation-config trtllm --no-telemetry
```

This two-GPU TEP and MTP3 path is validated with the complete checkpoint. The
resident PLE table was split into 160,000,768 BF16 rows (47.68 GiB) per rank and
completed concurrent batch-2 requests, long-context QSA, decode graph capture,
MTP3, and the deterministic GSM8K smoke. Validate the selected batch, sequence,
and cache capacities before using the larger values in this recipe as a
production SLA baseline. `allreduce_strategy` is intentionally omitted so that
the GB300 `AUTO` policy selects among supported collective implementations.

#### BF16 ADP2/EP2 with MTP3

This topology replicates attention computation over two token-owner ranks,
partitions routed experts with EP2, and row-shards the resident PLE table over
the attention-DP group. PLE exchanges global row IDs before lookup and
ReduceScatters BF16 embedding activations afterward. Save the validated
functionality configuration as `bf16-adp2-ep2-mtp3.yaml`:

```yaml
tensor_parallel_size: 2
moe_tensor_parallel_size: 1
moe_expert_parallel_size: 2
pipeline_parallel_size: 1
context_parallel_size: 1
enable_attention_dp: true
enable_lm_head_tp_in_adp: false
disable_mm_encoder: true

max_batch_size: 2
max_num_tokens: 2048
max_seq_len: 13312
enable_chunked_prefill: true
disable_overlap_scheduler: true

cuda_graph_config:
  enable_padding: true
  max_batch_size: 2

moe_config:
  backend: TRTLLM
  max_num_tokens: 2048
  disable_finalize_fusion: true

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  max_tokens: 32768
  avg_seq_len: 8192
  use_kv_cache_manager_v2: true
  enable_block_reuse: false

speculative_config:
  decoding_type: MTP
  max_draft_len: 3
```

Start the server with the same command as the TEP2 recipe, substituting this
configuration. The production-size smoke completed two simultaneous requests,
a 4,853-token QSA request, decode graph capture for batch sizes one and two,
MTP3, and GSM8K 8/8. It measured 172.02 GiB after weight loading and a
178.17-GiB profile peak per rank. Attention DP with context parallelism greater
than one is not supported by the PLE row-sharding collective.

Both two-GPU recipes use device-resident PLE shards by default. Set
`TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1` and the pinned allocator configuration
from the PLE validation section to move those rank-local shards to host memory.

#### BF16 TP4 with CUTLASS MoE

Use CUTLASS when pure MoE-TP4 produces a BF16 TRTLLM kernel shape that is not
eligible.

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 4
moe_expert_parallel_size: 1
pipeline_parallel_size: 1
enable_attention_dp: false
disable_mm_encoder: true

max_batch_size: 16
max_num_tokens: 8192
max_seq_len: 16384
enable_chunked_prefill: true
disable_overlap_scheduler: false

moe_config:
  backend: CUTLASS
  disable_finalize_fusion: true

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  use_kv_cache_manager_v2: true
  enable_block_reuse: false
```

`allreduce_strategy` is intentionally omitted so the production default remains
`AUTO`.

#### BF16 TEP4 with TRTLLM MoE

Keeping the routed-expert tensor-parallel size at one preserves the complete
640-element intermediate dimension on each expert owner.

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 1
moe_expert_parallel_size: 4
enable_attention_dp: false
disable_mm_encoder: true

max_batch_size: 16
max_num_tokens: 8192
max_seq_len: 16384
enable_chunked_prefill: true
disable_overlap_scheduler: false

moe_config:
  backend: TRTLLM
  disable_finalize_fusion: true

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  use_kv_cache_manager_v2: true
```

#### Block-FP8 TP1 with MTP3 on one GB300

TP1 avoids both expert sharding and FP8 scale-block ambiguity. The conservative
configuration below uses the validated TRTLLM MoE backend and the single-GPU
MTP3 envelope. Save it as `fp8-tp1-mtp3.yaml`:

```yaml
tensor_parallel_size: 1
moe_tensor_parallel_size: 1
moe_expert_parallel_size: 1
enable_attention_dp: false
disable_mm_encoder: true

max_batch_size: 1
max_num_tokens: 2048
max_seq_len: 13312
enable_chunked_prefill: true
disable_overlap_scheduler: true

cuda_graph_config:
  enable_padding: true
  max_batch_size: 1

moe_config:
  backend: TRTLLM
  max_num_tokens: 2048
  disable_finalize_fusion: true

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  max_tokens: 32768
  avg_seq_len: 4096
  use_kv_cache_manager_v2: true
  enable_block_reuse: false

speculative_config:
  decoding_type: MTP
  max_draft_len: 3
  use_rejection_sampling: true
  advanced_sampling_mode: full
```

Start the single-rank aggregate server with:

```bash
trtllm-llmapi-launch trtllm-serve /path/to/Qwen3.8-Flash-Next-FP8 \
  --config fp8-tp1-mtp3.yaml \
  --served_model_name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 8000 \
  --generation-config trtllm --no-telemetry
```

The complete block-FP8 checkpoint fits on one GB300 under the validated
functionality envelope, but remaining memory for KV cache and concurrent
requests is limited. Reduce `max_batch_size` or cache capacity if initialization
reports insufficient free memory. DeepGEMM is also validated on this topology,
but this copy-ready MTP3 example uses TRTLLM MoE to match the general block-FP8
serving path.

#### Block-FP8 TEP4

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 1
moe_expert_parallel_size: 4
enable_attention_dp: false
disable_mm_encoder: true

max_batch_size: 16
max_num_tokens: 8192
max_seq_len: 16384
enable_chunked_prefill: true
disable_overlap_scheduler: false

moe_config:
  backend: TRTLLM
  disable_finalize_fusion: true

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  use_kv_cache_manager_v2: true
```

### Feature overlays

#### MTP3

For greedy decoding, add the following block to a compatible BF16 or block-FP8
topology:

```yaml
speculative_config:
  decoding_type: MTP
  max_draft_len: 3
```

For non-greedy decoding, including requests with nonzero temperature or top-p
sampling, use distribution-correct rejection sampling:

```yaml
speculative_config:
  decoding_type: MTP
  max_draft_len: 3
  use_rejection_sampling: true
  advanced_sampling_mode: full
```

`advanced_sampling_mode: full` retains per-request top-k and top-p filtering.
The runtime detects non-greedy requests automatically. Do not use the deprecated
`allow_advanced_sampling` field; it is retained only as a no-op compatibility
field. The one-model rejection path requires a deployment image with compatible
FlashInfer components.

MTP acceptance depends on prompts, sampling, checkpoint, batch shape, and the
server configuration. An acceptance rate from one workload should not be
treated as a model constant. In particular, a short greedy smoke is not
comparable with a long thinking-enabled, temperature-1.0 workload.

#### CUDA graph, chunked prefill, overlap, and FP8 KV cache

```yaml
max_batch_size: 16
max_num_tokens: 8192
enable_chunked_prefill: true
disable_overlap_scheduler: false

cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16]

kv_cache_config:
  dtype: fp8
  use_kv_cache_manager_v2: true
  enable_block_reuse: false
```

For long prompts, choose `max_num_tokens` below the prompt length when the goal
is to exercise chunked prefill. Graph batch sizes should cover the expected
decode batch distribution rather than only the maximum batch size.

### Disaggregated text deployment recipes

#### BF16 TP4

The validated text configuration uses one TP4 context worker and one TP4
generation worker on separate four-GPU groups. Both workers use NIXL, KV cache
manager V2, QSA, and CUTLASS MoE. The context worker disables the overlap
scheduler because context-only disaggregated execution does not support it. KV
block reuse is disabled in this baseline so that state transfer, rather than a
local prefix-cache hit, is responsible for generation-worker reconstruction.

Use the following context-worker configuration as `context.yaml`:

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 4
moe_expert_parallel_size: 1
pipeline_parallel_size: 1
enable_attention_dp: false
disable_mm_encoder: true

max_batch_size: 1
max_input_len: 16384
max_seq_len: 17408
max_num_tokens: 16384
enable_chunked_prefill: false
disable_overlap_scheduler: true
cuda_graph_config: null

moe_config:
  backend: CUTLASS
  disable_finalize_fusion: true

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  max_tokens: 32768
  avg_seq_len: 8192
  enable_block_reuse: false
  use_kv_cache_manager_v2: true

cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  max_tokens_in_buffer: 16384
  kv_cache_bounce_size_mb: 1024

internal_request_auth_key: "replace-with-a-random-shared-key"
```

Use the following generation-worker configuration as `generation.yaml`:

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 4
moe_expert_parallel_size: 1
pipeline_parallel_size: 1
enable_attention_dp: false
disable_mm_encoder: true

max_batch_size: 8
max_input_len: 16384
max_seq_len: 17408
max_num_tokens: 2048
enable_chunked_prefill: false
disable_overlap_scheduler: true
cuda_graph_config: null

moe_config:
  backend: CUTLASS
  disable_finalize_fusion: true

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  max_tokens: 32768
  avg_seq_len: 8192
  enable_block_reuse: false
  use_kv_cache_manager_v2: true

cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  max_tokens_in_buffer: 16384
  kv_cache_bounce_size_mb: 1024

internal_request_auth_key: "replace-with-the-same-random-shared-key"
```

The proxy configuration, `disagg.yaml`, names the externally visible model and
the two worker endpoints:

```yaml
hostname: 0.0.0.0
port: 8000
model: Qwen3.8-Flash-Next
backend: pytorch
internal_request_auth_key: "replace-with-the-same-random-shared-key"
context_servers:
  num_instances: 1
  urls:
    - "context-host:8001"
generation_servers:
  num_instances: 1
  urls:
    - "generation-host:8002"
```

Generate a fresh authentication key for each deployment, inject the same value
into all three runtime configurations, and do not commit it. Start the workers
before the proxy:

```bash
# Run on the context worker's four-GPU group.
trtllm-llmapi-launch trtllm-serve /path/to/checkpoint \
  --config context.yaml \
  --served_model_name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 8001 \
  --generation-config trtllm --no-telemetry

# Run on the generation worker's separate four-GPU group.
trtllm-llmapi-launch trtllm-serve /path/to/checkpoint \
  --config generation.yaml \
  --served_model_name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 8002 \
  --generation-config trtllm --no-telemetry

# Start only after both worker health endpoints report ready.
trtllm-serve disaggregated -c disagg.yaml
```

The context and generation workers must use the same checkpoint revision and a
compatible model-state layout. `max_tokens_in_buffer` should be at least the
largest supported input length. In addition to ordinary attention KV blocks,
the model integration transfers QSA index state, Gated DeltaNet state, and PLE
n-gram and short-convolution side state.

#### Block-FP8 ADP4 with MTP3

The validated long-generation state-transfer profile uses attention data
parallelism and expert parallelism on both four-GPU workers. This preserves
routed-expert block-scale layout while allowing each attention rank to own its
recurrent state. The context worker does not run speculative decoding. The
generation worker runs MTP with a maximum draft length of three and padded
decode CUDA graphs. The rejection-sampling fields below are the required update
for distribution-correct non-greedy sampling; that sampler contract was
validated independently in aggregate TEP4 and ADP4 runs.

Use these material context-worker settings in addition to the NIXL and shared
authentication settings from the preceding example:

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 1
moe_expert_parallel_size: 4
enable_attention_dp: true
enable_lm_head_tp_in_adp: true
disable_mm_encoder: true

max_batch_size: 16
max_input_len: 8192
max_seq_len: 73728
max_num_tokens: 8192
enable_chunked_prefill: false
disable_overlap_scheduler: true
cuda_graph_config: null

moe_config:
  backend: TRTLLM
  disable_finalize_fusion: true

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  max_tokens: 1179648
  avg_seq_len: 32768
  enable_block_reuse: false
  use_kv_cache_manager_v2: true

cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  max_tokens_in_buffer: 8192
  kv_transfer_timeout_ms: 600000
  kv_cache_bounce_size_mb: 1024
```

Use the same topology and cache settings on the generation worker, with these
generation-specific changes:

```yaml
max_batch_size: 16
max_input_len: 8192
max_seq_len: 73728
max_num_tokens: 2048

cuda_graph_config:
  enable_padding: true
  max_batch_size: 16

speculative_config:
  decoding_type: MTP
  max_draft_len: 3
  use_rejection_sampling: true
  advanced_sampling_mode: full
```

`allreduce_strategy` is intentionally omitted in this profile so that the
validated SM103 `AUTO` policy is used. The overlap scheduler and chunked
prefill are disabled in this conservative disaggregated accuracy profile to
isolate state transfer and long-generation correctness; both features are
validated in aggregated serving and this setting is not a statement that they
are unsupported. For long generations, set the proxy request timeout above the
maximum expected request duration. The validation profile used 21,600 seconds;
the NIXL KV-transfer timeout was 600,000 milliseconds.

### Aggregate multimodal deployment recipes

#### BF16 TP4

The following conservative BF16 TP4 configuration is validated for aggregate
image-language serving with a local multimodal encoder on NVIDIA GB300 GPUs.
Save it as `multimodal-bf16-tp4.yaml`:

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 4
moe_expert_parallel_size: 1
enable_attention_dp: false
disable_mm_encoder: false

max_batch_size: 8
max_input_len: 4096
max_seq_len: 16384
max_num_tokens: 32768
encoder_max_batch_size: 8
encoder_max_num_tokens: 65536
enable_chunked_prefill: false
disable_overlap_scheduler: true
cuda_graph_config: null

moe_config:
  backend: CUTLASS
  disable_finalize_fusion: true

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  max_tokens: 65536
  enable_block_reuse: false
  use_kv_cache_manager_v2: true
```

#### Block-FP8 TP1

For the pre-quantized block-FP8 checkpoint, use the following self-contained
configuration on one GB300 backend GPU. TP1 keeps each routed expert and its
128-by-128 scale grid local; the checkpoint's vision modules remain
unquantized. Save it as `multimodal-fp8-tp1.yaml`:

```yaml
tensor_parallel_size: 1
moe_tensor_parallel_size: 1
moe_expert_parallel_size: 1
pipeline_parallel_size: 1
enable_attention_dp: false
disable_mm_encoder: false

max_batch_size: 8
max_input_len: 4096
max_seq_len: 16384
max_num_tokens: 8192
encoder_max_batch_size: 8
encoder_max_num_tokens: 65536
enable_chunked_prefill: false
disable_overlap_scheduler: true
cuda_graph_config: null

moe_config:
  backend: TRTLLM
  max_num_tokens: 8192
  disable_finalize_fusion: true

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  max_tokens: 65536
  enable_block_reuse: false
  use_kv_cache_manager_v2: true
```

Start an aggregate server with the multimodal-disaggregation switch explicitly
disabled. Setting the value explicitly prevents an inherited environment value
from selecting the unsupported encoder-handoff path. Select the configuration
and matching checkpoint precision from the preceding examples:

```bash
TLLM_MULTIMODAL_DISAGGREGATED=0 \
trtllm-llmapi-launch trtllm-serve /path/to/checkpoint \
  --config multimodal-bf16-tp4.yaml \
  --served_model_name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 8000 \
  --generation-config trtllm --no-telemetry
```

Use `multimodal-fp8-tp1.yaml` instead when serving the block-FP8 checkpoint.

An OpenAI-compatible single-image request uses an `image_url` content part
followed by the text instruction:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        {"type": "text", "text": "Describe the image briefly."}
      ]
    }],
    "temperature": 0.0,
    "max_tokens": 128
  }'
```

The image URL must be reachable from the serving frontend; a supported data URL
may be used instead. Add another `image_url` content part for a multi-image
request and preserve content-part order. Do not set `disable_mm_encoder: true`
for image-language requests. QSA, the selected MoE topology, and KV cache manager
V2 still govern the text decoder.

The retained acceptance smoke used four single-image cases and one ordered
two-image case. The five cases ran first sequentially and then concurrently for
10 total requests per checkpoint precision. Every request returned HTTP 200,
non-empty schema-valid output, the expected semantic result, and a normal stop
reason. Server logs also confirmed QSA execution, KV cache manager V2,
`is_disagg=False`, and a disabled cache transceiver. BF16 TP4 and block-FP8 TP1
both passed this contract.

## Operational notes

- Initial startup can include checkpoint loading, kernel selection, CUDA graph
  capture, and compilation or cache population. Preserve writable persistent
  caches and measure steady-state service performance only after initialization
  completes.
- A response that reaches the configured maximum-generation limit remains in
  the accuracy denominator. It is an output-limit condition, not a transport or
  request-integrity failure.
- The preceding multimodal results validate BF16 TP4 and block-FP8 TP1 aggregate
  serving with a local encoder. They do not validate a separate multimodal
  encoder-to-prefill handoff.
- Feature-coexistence checks establish correctness, not an isolated latency or
  throughput benefit. Performance sign-off requires a fixed workload and
  repeated measurements on the target deployment.

## Implementation references

- Model configuration and registration:
  [`qwen4_exp.py`](../../../tensorrt_llm/_torch/configs/qwen4_exp.py) and
  [`modeling_qwen4_exp.py`](../../../tensorrt_llm/_torch/models/modeling_qwen4_exp.py)
- Multimodal wrapper and shared vision-language input handling:
  [`modeling_qwen4_exp.py`](../../../tensorrt_llm/_torch/models/modeling_qwen4_exp.py),
  [`modeling_qwen3vl.py`](../../../tensorrt_llm/_torch/models/modeling_qwen3vl.py), and
  [`modeling_multimodal_mixin.py`](../../../tensorrt_llm/_torch/models/modeling_multimodal_mixin.py)
- QSA model integration and sparse backend:
  [`modeling_qwen4_exp_attention.py`](../../../tensorrt_llm/_torch/models/modeling_qwen4_exp_attention.py)
  and [`qsa/`](../../../tensorrt_llm/_torch/attention_backend/sparse/qsa/)
- Block-FP8 checkpoint mapping:
  [`qwen4_exp_weight_mapper.py`](../../../tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py)
- Hyper-Connections and PLE:
  [`qwen4_exp_hyper_connection.py`](../../../tensorrt_llm/_torch/modules/qwen4_exp_hyper_connection.py)
  and [`qwen4_exp_ple.py`](../../../tensorrt_llm/_torch/modules/qwen4_exp_ple.py)
- MoE backend resolution and kernels:
  [`moe_resolution.py`](../../../tensorrt_llm/_torch/modules/fused_moe/moe_resolution.py),
  [`fused_moe_trtllm_gen.py`](../../../tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py),
  [`fused_moe_cutlass.py`](../../../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py), and
  [`fused_moe_deepgemm.py`](../../../tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py)
- MTP and accepted-state lifecycle:
  [`modeling_speculative.py`](../../../tensorrt_llm/_torch/models/modeling_speculative.py),
  [`mtp.py`](../../../tensorrt_llm/_torch/speculative/mtp.py), and
  [`interface.py`](../../../tensorrt_llm/_torch/speculative/interface.py)
- Recurrent cache and disaggregated state transfer:
  [`mamba_cache_manager.py`](../../../tensorrt_llm/_torch/pyexecutor/mamba_cache_manager.py),
  [`page.py`](../../../tensorrt_llm/_torch/disaggregation/resource/page.py), and
  [`kv_extractor.py`](../../../tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py)

## Release boundary

This matrix is a functionality-support statement. It does not publish a latency,
throughput, capacity, or service-level guarantee. Deployment-specific performance
sign-off requires fixed request distributions, concurrency, input/output lengths,
warmup policy, cache state, and repeated measured runs on the target hardware.
