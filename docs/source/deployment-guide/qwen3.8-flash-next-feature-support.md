<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-Flash-Next Feature Support and Deployment

## Overview

This document describes TensorRT-LLM PyTorch-backend support for
Qwen3.8-Flash-Next BF16, pre-quantized block-FP8, and NVIDIA mixed-precision
NVFP4 checkpoints. The Hugging Face conditional-generation architecture is
`Qwen4ExpForConditionalGeneration`; language-only serving resolves the decoder
as `Qwen4ExpForCausalLM`.

The model combines QSA sparse full-attention layers, Gated DeltaNet recurrent
layers, Hyper-Connections, PLE recurrent state, and routed and shared experts.
These components have state and parallelism requirements beyond those of a
conventional decoder-only transformer and require model-specific state ownership
throughout prefill, decode, cache reuse, and disaggregated serving.

The NVIDIA NVFP4 export is intentionally described as mixed precision: its main
routed experts use NVFP4, while the PLE n-gram table and MTP experts use FP8.
TensorRT-LLM therefore reports `MIXED_PRECISION`, rather than plain `NVFP4`, for
this checkpoint. The weight format does not alter the request-state transfer
layout used by disaggregated serving.

### Official checkpoints

The official Hugging Face checkpoints are:

- BF16: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Block-FP8: [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
- NVIDIA mixed-precision NVFP4: use the NVIDIA-provided
  Qwen3.8-Flash-Next NVFP4 checkpoint.

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
| Hyper-Connections | **Validated** | **Validated** | Multi-stream residual mixing and topology-specific reductions are part of the accepted end-to-end paths. CUDA execution fuses gated stream reduction, residual injection, and the combine-to-grouped-RMSNorm boundary. |
| PLE | **Validated** | **Validated** | Token and short-convolution state are managed per request and participate in chunk continuation, prefix reuse, and MTP accepted-state promotion. The n-gram table is row-sharded across TP ranks; ADP preserves rank-local token ownership with gather and reduce-scatter collectives. |
| QSA/GDN/PLE state with KV cache manager V2 | **Validated** | **Validated** | KV cache manager V2 is the required lifecycle owner for the model-specific auxiliary state. |

### Parallelism, communication, and speculative decoding

| Feature | BF16 checkpoint | Block-FP8 checkpoint | Notes |
|---|---|---|---|
| Tensor parallelism | **Validated** | **Validated with constraints** | BF16 TP4 is validated. Block-FP8 TP1 is validated on one GB300. For block-FP8, routed-expert tensor-parallel partitions must preserve 128-element scale blocks. |
| Tensor/expert parallelism (TEP) | **Validated** | **Validated** | TEP4 is validated for both precisions. Block-FP8 also has a validated two-GPU TEP topology. Use `moe_tensor_parallel_size: 1` and expert parallelism when a pure MoE-TP split would cut a scale block. |
| Attention data parallelism (ADP) | **Validated** | **Validated** | ADP4 is validated with expert-parallel routed experts and model-specific recurrent-state ownership. Small topology-sensitive BF16 numerical differences are expected and are not treated as request-integrity failures. |
| Pipeline parallelism | **Validated** | **Not validated** | BF16 PP4 and PP2+TP2 functional paths are validated. No block-FP8 PP release claim is made by this matrix. |
| MTP with draft depth 3 | **Validated** | **Validated** | The recurrent MTP layer supports a configured maximum draft length of three, including accepted-prefix promotion of GDN, QSA, and PLE state. Greedy decoding uses strict acceptance; distribution-correct non-greedy decoding uses rejection sampling with the full advanced sampler. Other draft depths are not part of this release claim. |
| GDN replay under MTP | **Validated** | **Validated** | Replay and accepted-state commit are covered together with semantic output checks. |

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
| Text prefill/decode disaggregated serving | **Supported** | **Validated** | NIXL disaggregation transfers attention KV pages, QSA index state, Gated DeltaNet state, and replicated PLE n-gram and convolution state. The block-FP8 TP1 path has production-checkpoint semantic coverage; the documented BF16 TP4 topology still needs an equivalent current-revision run. Context and generation workers must use compatible checkpoint and state layouts. This claim does not include a separate multimodal encoder-to-prefill handoff. |

### NVIDIA mixed-precision NVFP4 checkpoint

The NVIDIA checkpoint loader supports the mixed quantization metadata used by
this export: NVFP4 main-model routed experts, an FP8 PLE n-gram table, and FP8
block-scaled MTP experts. Aggregate one-GPU text and multimodal MTP3 accuracy
tests cover the TRTLLM MoE backend with PLE host offload.

Disaggregated serving transfers request state, not model weights. The context
and generation workers each load the complete same checkpoint, while NIXL
transfers attention KV pages, QSA index state, Gated DeltaNet state, the PLE
short-convolution state, and the integer PLE n-gram context. Consequently, no
NVFP4-specific wire format is required. Production-size disaggregated semantic
validation of this exact checkpoint remains separate from the BF16 and
block-FP8 evidence below; do not treat aggregate checkpoint-loading tests as
that evidence.

## Validation evidence

The results below are retained pre-reconstruction evidence. They report complete
datasets or explicit smoke-test contracts, preserve request failures and
generation-limit responses in the denominator, and combine client-side semantic
checks with server-side runtime evidence where applicable.

These measurements used TensorRT-LLM source `d6873e2426`. The BF16 weight index
matches Hugging Face revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`, and
the block-FP8 weight index matches revision
`236dfdf285828023ca3bcd3f37366c58a3469b13`. They are prior validation evidence,
not measurements of later source revisions. Validation of a later revision
must be recorded separately.

For reproducibility, retain the resolved server configuration and final LLM
arguments, individual responses and finish reasons, per-request performance
records, and backend-resolution logs. A configured feature is not considered
verified unless the effective runtime state and request records show that it ran.

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

### NVIDIA mixed-precision NVFP4 accuracy

The NVIDIA mixed-precision checkpoint was evaluated on one B200 with MTP3,
TRTLLM MoE, and PLE host offload. The text path scored 95.83 on GSM8K and
87.60 on MMLU; the aggregate multimodal path scored 58.56 on MMMU. These
results validate checkpoint loading and aggregate inference. They do not, by
themselves, validate a context-to-generation state transfer.

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
| PLE table sharding | BF16 TEP4 and ADP4 loaded only the locally owned n-gram embedding rows and passed semantic checks. | Four-way sharding avoids approximately 71.5 GiB of replicated PLE weight storage per rank. |
| MTP state lifecycle | Draft depth 3, GDN replay, and accepted-prefix promotion of QSA, GDN, and PLE state were observed with semantic output checks. | MTP3 state promotion is validated; other draft depths are not claimed. |
| Combined serving features | CUDA graph padding, chunked prefill, overlap scheduling, QSA, and FP8 KV cache ran together with semantic checks. | Feature coexistence is validated; no isolated performance gain is claimed. |

## Deployment constraints

### Block-FP8 checkpoint format

The block-FP8 checkpoint is pre-quantized. It is not produced by setting an FP8
option while loading the BF16 checkpoint. Routed-expert projections carry FP8
weights and inverse scales with 128-by-128 blocks, while modules excluded by the
checkpoint quantization metadata remain in their original precision. The runtime
must preserve that metadata during weight mapping and must select an MoE backend
that consumes `FP8_BLOCK_SCALES`.

### NVIDIA mixed-precision NVFP4 checkpoint format

Do not override this checkpoint to a single global quantization algorithm. Its
per-layer `quant_config_dict` is part of the checkpoint contract: main-model
routed experts use NVFP4, the PLE n-gram table uses scaled FP8, and MTP experts
use FP8 block scales. TensorRT-LLM normalizes both main-model and MTP layer names
before applying those entries. Use the TRTLLM MoE backend for the documented
NVIDIA checkpoint path.

The PLE table is much larger than the per-request PLE state. Enabling PLE host
offload keeps the table in pinned host memory, but it does not move the
request's short-convolution state or n-gram context out of the cache manager.
Those request states remain part of every disaggregated handoff.

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

The same separation applies to the mixed-precision NVFP4 checkpoint: model
weight precision does not select the attention KV-cache dtype or the Gated
DeltaNet recurrent-state dtype.

## Example deployment recipes

The examples below are functionality-oriented, end-to-end starting points. They
cover the primary aggregate deployment shapes: BF16 TEP2 with MTP3, block-FP8
TP1 with MTP3, and aggregate multimodal serving. Capacity fields must be sized
for the selected checkpoint, GPU count, input/output lengths, and concurrency.
Performance-sensitive deployments should tune from a matched validated baseline.

### Choosing a recipe

| Deployment goal | Starting point | Validation status |
|---|---|---|
| BF16 on two GB300 GPUs with MTP3 | BF16 TEP2 with MTP3 | Supported starting point; constituent TP2 MTP3 and TEP4 MTP3 paths are validated |
| Block-FP8 on one GB300 with MTP3 | Block-FP8 TP1 with MTP3 | Validated |
| BF16 tensor-parallel text serving | BF16 TP4 with CUTLASS MoE | Validated |
| BF16 expert-parallel text serving | BF16 TEP4 with TRTLLM MoE | Validated |
| Block-FP8 multi-GPU text serving | Block-FP8 TEP4 with TRTLLM MoE | Validated |
| NVIDIA mixed-precision NVFP4 on one Blackwell GPU | Reuse the block-FP8 TP1 topology with TRTLLM MoE and PLE host offload | Aggregate loading and accuracy validated; disaggregated validation pending |
| MTP3 | Add the greedy or non-greedy MTP3 overlay | Validated |
| Prefix caching | Use the complete cache configuration in the Validation evidence section | Validated |
| CUDA graph, chunked prefill, overlap, and FP8 KV cache | Add the combined feature overlay | Validated for coexistence |
| Text prefill/decode disaggregation | BF16 TP4 or block-FP8 ADP4 context and generation workers | Block-FP8 validated with NIXL and KV cache manager V2; BF16 supported, current-revision validation pending |
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
allreduce_strategy: NCCL
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

This is the two-GPU form of the supported TEP and MTP3 paths. Retained
production-size evidence separately covers BF16 TP2 with MTP3 and BF16 TEP4
with MTP3. Validate the selected batch, sequence, and cache capacities before
using this TEP2 recipe as a production SLA baseline. The recipe pins NCCL so it
does not depend on platform-specific automatic collective selection.

#### BF16 TP4 with CUTLASS MoE

Use CUTLASS when pure MoE-TP4 produces a BF16 TRTLLM kernel shape that is not
eligible.

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 4
moe_expert_parallel_size: 1
pipeline_parallel_size: 1
enable_attention_dp: false
allreduce_strategy: NCCL
disable_mm_encoder: true

max_batch_size: 16
max_num_tokens: 8192
max_seq_len: 16384
enable_chunked_prefill: true
disable_overlap_scheduler: false

moe_config:
  backend: CUTLASS

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  use_kv_cache_manager_v2: true
  enable_block_reuse: false
```

`allreduce_strategy: NCCL` is explicit in this standalone recipe.

#### BF16 TEP4 with TRTLLM MoE

Keeping the routed-expert tensor-parallel size at one preserves the complete
640-element intermediate dimension on each expert owner.

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 1
moe_expert_parallel_size: 4
enable_attention_dp: false
allreduce_strategy: NCCL
disable_mm_encoder: true

max_batch_size: 16
max_num_tokens: 8192
max_seq_len: 16384
enable_chunked_prefill: true
disable_overlap_scheduler: false

moe_config:
  backend: TRTLLM

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

#### NVIDIA mixed-precision NVFP4 with MTP3

Use the preceding TP1 configuration with the NVIDIA NVFP4 checkpoint and keep
`moe_config.backend: TRTLLM`. Enable PLE host offload before starting the
server:

```bash
export TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1
trtllm-llmapi-launch trtllm-serve /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --config fp8-tp1-mtp3.yaml \
  --served_model_name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 8000 \
  --generation-config trtllm --no-telemetry
```

The file name `fp8-tp1-mtp3.yaml` describes the topology inherited from the
preceding recipe; it does not override the NVFP4 checkpoint's mixed per-layer
quantization metadata.

#### Block-FP8 TEP4

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 1
moe_expert_parallel_size: 4
enable_attention_dp: false
allreduce_strategy: NCCL
disable_mm_encoder: true

max_batch_size: 16
max_num_tokens: 8192
max_seq_len: 16384
enable_chunked_prefill: true
disable_overlap_scheduler: false

moe_config:
  backend: TRTLLM

sparse_attention_config:
  algorithm: qsa

kv_cache_config:
  use_kv_cache_manager_v2: true
```

#### MoE finalize fusion and repeatability

BF16 CUTLASS MoE uses the fused FC2 finalization path by default. At the
model's 512-expert, top-10 routing shape, this removes a standalone routing
finalization launch and improved CUDA-graph MoE latency across the validated
decode batch sizes. Aggregate BF16 TEP4 semantic validation passed with the
fusion enabled.

The fused parallel reduction is numerically valid but is not bitwise
deterministic. Small BF16 rounding differences can therefore change a later
greedy token. Deployments that require exact run-to-run output replay may set
`moe_config.disable_finalize_fusion: true`; doing so uses the separate
deterministic FP32 finalization path and can reduce performance.

This option controls the CUTLASS BF16 path. The validated block-FP8 TRTLLM and
DeepGEMM backends use their own finalization paths and do not require this
setting.

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

### Text prefill/decode disaggregated serving

Disaggregated text serving uses a context worker for prefill, a generation
worker for decode, and a proxy that presents the external endpoint. Both
workers must use the same checkpoint revision and compatible model-state
layouts. KV cache manager V2 is required because the transferred request state
includes attention KV pages, QSA index state, Gated DeltaNet state, and PLE
n-gram and short-convolution state.

The examples below use NIXL and explicitly select NCCL collectives. Generate a
fresh shared authentication key for every deployment, provide the same value to
both workers and the proxy, and do not commit the key. Start both workers and
wait for their health endpoints before starting the proxy.

#### BF16 TP4 context worker

Save the following as `context.yaml` and run it on a four-GPU context group:

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 4
moe_expert_parallel_size: 1
pipeline_parallel_size: 1
enable_attention_dp: false
allreduce_strategy: NCCL
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

internal_request_auth_key: "replace-with-a-random-shared-key"
```

#### BF16 TP4 generation worker

Save the following as `generation.yaml` and run it on a separate four-GPU
generation group:

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 4
moe_expert_parallel_size: 1
pipeline_parallel_size: 1
enable_attention_dp: false
allreduce_strategy: NCCL
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

internal_request_auth_key: "replace-with-the-same-random-shared-key"
```

#### Block-FP8 ADP4 with generation-side MTP3

For block-FP8, use expert parallelism so each routed expert retains its complete
128-by-128 scale grid. The material context-worker settings are:

```yaml
tensor_parallel_size: 4
moe_tensor_parallel_size: 1
moe_expert_parallel_size: 4
enable_attention_dp: true
enable_lm_head_tp_in_adp: true
allreduce_strategy: NCCL
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
  kv_transfer_timeout_ms: 600000

internal_request_auth_key: "replace-with-a-random-shared-key"
```

Use the same topology, cache manager, transceiver, and authentication settings
on the generation worker, with these generation-specific overrides:

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

The context worker does not run speculative decoding. Rejection sampling with
the full advanced sampler is required for distribution-correct non-greedy
generation. The conservative disaggregated accuracy recipe disables chunked
prefill and overlap scheduling to isolate state transfer; their aggregate
support is documented separately.

#### NVIDIA mixed-precision NVFP4

The disaggregated state protocol is independent of the model-weight format, so
the NVIDIA mixed-precision checkpoint uses the same NIXL and KV cache manager V2
settings. Both workers must load the same checkpoint revision, select
`moe_config.backend: TRTLLM`, and use the same PLE host-offload setting. For a
one-GPU-per-worker starting point, change the topology fields in both worker
configs to:

```yaml
tensor_parallel_size: 1
moe_tensor_parallel_size: 1
moe_expert_parallel_size: 1
enable_attention_dp: false

moe_config:
  backend: TRTLLM
```

Start each worker with host offload enabled:

```bash
export TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=1
```

Keep `speculative_config` only on the generation worker. This configuration is
a functional starting point derived from the aggregate mixed-precision loader
and the precision-independent state-transfer contract; qualify its memory
capacity and end-to-end semantics on the target hardware before treating it as
a deployment baseline.

#### Proxy and launch commands

Save the proxy configuration as `disagg.yaml`:

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

Start the matching checkpoint on each worker, then start the proxy:

```bash
# Context worker GPU group.
trtllm-llmapi-launch trtllm-serve /path/to/checkpoint \
  --config context.yaml \
  --served_model_name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 8001 \
  --generation-config trtllm --no-telemetry

# Separate generation worker GPU group.
trtllm-llmapi-launch trtllm-serve /path/to/checkpoint \
  --config generation.yaml \
  --served_model_name Qwen3.8-Flash-Next \
  --host 0.0.0.0 --port 8002 \
  --generation-config trtllm --no-telemetry

# Start after both worker health endpoints report ready.
trtllm-serve disaggregated -c disagg.yaml
```

Leave `max_tokens_in_buffer` unset unless the deployment needs an explicit
admission limit; its default is derived from the model's maximum sequence
length. A positive `kv_cache_bounce_size_mb` can coalesce scattered transfers
on systems with fabric memory, but it is hardware- and payload-size-dependent
and is therefore not part of the portable baseline above. For long generations,
also set the proxy request timeout above the expected request duration. These
recipes validate state-transfer functionality; context/generation capacity
matching and disaggregated throughput tuning are separate deployment tasks.

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
allreduce_strategy: NCCL
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
  serving with a local encoder. A separate encoder-to-prefill handoff is outside
  this guide's support boundary.
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
  [`modeling_qwen4_exp.py`](../../../tensorrt_llm/_torch/models/modeling_qwen4_exp.py)
  and [`qsa/`](../../../tensorrt_llm/_torch/attention/backends/sparse/qsa/)
- Block-FP8 and mixed-precision NVFP4 checkpoint mapping:
  [`qwen4_exp_weight_mapper.py`](../../../tensorrt_llm/_torch/models/checkpoints/hf/qwen4_exp_weight_mapper.py)
- Hyper-Connections and PLE:
  [`hyper_connection.py`](../../../tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection.py),
  [`hyper_connection_kernels.py`](../../../tensorrt_llm/_torch/modules/qwen4_exp/hyper_connection_kernels.py),
  and [`ple.py`](../../../tensorrt_llm/_torch/modules/qwen4_exp/ple.py)
- MoE backend resolution and kernels:
  [`moe_resolution.py`](../../../tensorrt_llm/_torch/moe/fused_moe/moe_resolution.py),
  [`fused_moe_trtllm_gen.py`](../../../tensorrt_llm/_torch/moe/fused_moe/fused_moe_trtllm_gen.py),
  [`fused_moe_cutlass.py`](../../../tensorrt_llm/_torch/moe/fused_moe/fused_moe_cutlass.py), and
  [`fused_moe_deepgemm.py`](../../../tensorrt_llm/_torch/moe/fused_moe/fused_moe_deepgemm.py)
- MTP and accepted-state lifecycle:
  [`modeling_speculative.py`](../../../tensorrt_llm/_torch/models/modeling_speculative.py),
  [`mtp.py`](../../../tensorrt_llm/_torch/speculative/mtp.py), and
  [`interface.py`](../../../tensorrt_llm/_torch/speculative/interface.py)
- Recurrent cache lifecycle:
  [`mamba_cache_manager.py`](../../../tensorrt_llm/_torch/pyexecutor/kv_cache/mamba_cache_manager.py)
- Text disaggregated state description and transfer:
  [`kv_extractor.py`](../../../tensorrt_llm/_torch/disaggregation/resource/kv_extractor.py),
  [`peer.py`](../../../tensorrt_llm/_torch/disaggregation/native/mixers/ssm/peer.py), and
  [`transceiver.py`](../../../tensorrt_llm/_torch/disaggregation/transceiver.py)

## Release boundary

This matrix is a functionality-support statement. It does not publish a latency,
throughput, capacity, or service-level guarantee. Deployment-specific performance
sign-off requires fixed request distributions, concurrency, input/output lengths,
warmup policy, cache state, and repeated measured runs on the target hardware.
