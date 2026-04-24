<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# DeepSeek V4 Flash AutoDeploy Support Plan

This document lays out a performance-oriented plan for serving
`deepseek-ai/DeepSeek-V4-Flash` through AutoDeploy. It focuses on the model
architecture, checkpoint quantization format, AutoDeploy cache/runtime
constraints, custom kernels, CUDA graph capture, and validation order.

The plan assumes:

- The modeling scaffold may be customized for AutoDeploy.
- AutoDeploy graph transforms may be extended.
- Custom Triton, TRT-LLM, or DeepGEMM-style kernels may be added.
- The initial target is reliable prefill/decode serving through AutoDeploy,
  with CUDA graph support.
- MTP support is not required for the first production path.

## Executive Summary

DeepSeek V4 Flash should not be treated as a normal FP8 model, and it should
not be forced into the existing classic DeepSeek V3 MLA path.

The deployment should introduce two DeepSeek V4-specific AutoDeploy surfaces:

1. `torch_deepseek_v4_sparse_attention`
   - Canonical source op for heterogeneous attention.
   - Lowered to sparse cached attention kernels with V4-specific paged cache
     resources.

2. `torch_deepseek_v4_moe`
   - Canonical source op for the V4 router and MoE block.
   - Lowered to fused routed+shared MoE kernels using packed MXFP4/FP4 expert
     weights.

The checkpoint is mixed precision:

- Attention, output projections, and shared expert linears are FP8 E4M3 with
  E8M0 block scales.
- Routed expert weights are packed FP4 stored in `I8` tensors, with E8M0
  scales.
- Router, compressor, embeddings/head, norms, hyper-connection tensors, and
  attention sinks are BF16/F32/I64.

Generic AutoDeploy `quant_method: fp8` handling is therefore insufficient.
DeepSeek V4 needs a model-specific mixed-quantization bridge that routes
ordinary linears to FineGrained FP8 and routed experts to MXFP4.

## Evidence Collected

The visible `HF_HOME` in the current environment points to:

```text
/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev/hf_home
```

That cache visibly contains `deepseek-ai/DeepSeek-V3`, but not
`deepseek-ai/DeepSeek-V4-Flash`. To avoid relying on guesswork, the following
small public metadata files were inspected directly:

- `config.json`
- `inference/config.json`
- `model.safetensors.index.json`
- selected safetensors headers from shards `00002`, `00005`, and `00045`

No full DeepSeek V4 weight download is required to identify the tensor dtypes,
shapes, and checkpoint naming convention.

Primary sources:

- Hugging Face model: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
- Hugging Face config:
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json>
- Hugging Face reference implementation:
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/inference/model.py>
- vLLM blog: <https://vllm.ai/blog/deepseek-v4>
- vLLM support PR: <https://github.com/vllm-project/vllm/pull/40760>

## Model Architecture Facts

From the DeepSeek V4 Flash config:

| Field | Value |
| - | - |
| Model type | `deepseek_v4` |
| Architecture | `DeepseekV4ForCausalLM` |
| Total layers | 43 |
| Hidden size | 4096 |
| Attention heads | 64 |
| KV heads | 1 |
| Head dim | 512 |
| Q LoRA rank | 1024 |
| QK RoPE head dim | 64 |
| Output groups | 8 |
| Output LoRA rank | 1024 |
| Sliding window | 128 |
| Routed experts | 256 |
| Experts per token | 6 |
| Shared experts | 1 |
| Hash-routed layers | 3 |
| MoE intermediate size | 2048 |
| Router scoring | `sqrtsoftplus` |
| Router scale | 1.5 |
| SwiGLU limit | 10.0 |
| Max position embeddings | 1,048,576 |

The `compress_ratios` pattern is:

```text
layer 0:  0
layer 1:  0
layer 2:  4
layer 3:  128
layer 4:  4
layer 5:  128
...
layer 40: 4
layer 41: 128
layer 42: 4
layer 43 entry in config list: 0
```

For the 43 actual transformer layers, the practical pattern is:

- first two layers: uncompressed attention
- middle layers: alternating ratio-4 and ratio-128 compression
- final listed ratio includes an uncompressed sentinel-style entry in the config
  list

The reference implementation uses:

- Sparse/local window attention over the last 128 tokens.
- Multi-head compression (MHC) over older context.
- Ratio-4 compressed layers with an indexer top-k path.
- Ratio-128 compressed layers without the same high-cost indexer path.
- Attention sink values per head.
- Hyper-connections around attention and MoE.
- One MTP block in the checkpoint/reference, but normal serving can initially
  omit MTP.

## Checkpoint Anatomy

The safetensors index reports:

```text
total_size: 159,609,485,896 bytes
num_tensors: 69,187
num_shards: 46
```

Tensor suffix counts:

```text
34,600  weight
34,167  scale
    62  ape
    44  hc_ffn_scale
    44  hc_ffn_fn
    44  hc_ffn_base
    44  hc_attn_scale
    44  hc_attn_fn
    44  hc_attn_base
    44  attn_sink
    41  bias
     3  tid2eid
```

Scale tensor categories:

```text
33,792  ffn.experts
   241  attn
   132  ffn.shared_experts
     2  other
```

The routed expert scale count matches:

```text
44 MoE-like blocks * 256 experts * 3 projections = 33,792
```

The extra MoE-like block is `mtp.0`.

### Representative Tensor Headers

Selected safetensors headers show the exact mixed checkpoint format:

```text
layers.0.attn.wq_a.weight   F8_E4M3  [1024, 4096]
layers.0.attn.wq_a.scale    F8_E8M0  [8, 32]

layers.0.attn.wq_b.weight   F8_E4M3  [32768, 1024]
layers.0.attn.wq_b.scale    F8_E8M0  [256, 8]

layers.0.attn.wkv.weight    F8_E4M3  [512, 4096]
layers.0.attn.wkv.scale     F8_E8M0  [4, 32]

layers.0.attn.wo_a.weight   F8_E4M3  [8192, 4096]
layers.0.attn.wo_a.scale    F8_E8M0  [64, 32]

layers.0.attn.wo_b.weight   F8_E4M3  [4096, 8192]
layers.0.attn.wo_b.scale    F8_E8M0  [32, 64]

layers.0.ffn.experts.0.w1.weight  I8       [2048, 2048]
layers.0.ffn.experts.0.w1.scale   F8_E8M0  [2048, 128]

layers.0.ffn.experts.0.w2.weight  I8       [4096, 1024]
layers.0.ffn.experts.0.w2.scale   F8_E8M0  [4096, 64]

layers.0.ffn.experts.0.w3.weight  I8       [2048, 2048]
layers.0.ffn.experts.0.w3.scale   F8_E8M0  [2048, 128]

layers.0.ffn.shared_experts.w1.weight  F8_E4M3  [2048, 4096]
layers.0.ffn.shared_experts.w1.scale   F8_E8M0  [16, 32]

layers.0.ffn.gate.weight    BF16  [256, 4096]
layers.0.ffn.gate.tid2eid   I64   [129280, 6]

layers.3.ffn.gate.weight    BF16  [256, 4096]
layers.3.ffn.gate.bias      F32   [256]

layers.3.attn.compressor.wkv.weight    BF16  [512, 4096]
layers.3.attn.compressor.wgate.weight  BF16  [512, 4096]

head.weight                 BF16  [129280, 4096]
norm.weight                 BF16  [4096]
```

### Quantization Interpretation

The checkpoint should be classified as:

| Component | Checkpoint dtype | Scale dtype | Runtime plan |
| - | - | - | - |
| Attention linears | `F8_E4M3` | `F8_E8M0` | FineGrained FP8 linear |
| Output low-rank projections | `F8_E4M3` | `F8_E8M0` | FineGrained FP8 linear |
| Shared expert linears | `F8_E4M3` | `F8_E8M0` | FineGrained FP8 linear or fused shared expert |
| Routed expert linears | packed FP4 in `I8` | `F8_E8M0` | MXFP4/FP4 fused MoE |
| Router weights | `BF16` | none | BF16 router kernel |
| Hash route table | `I64` | none | gather-based routing table |
| Compressor linears | `BF16` | none | BF16, later optionally quantized only after validation |
| Embedding/head | `BF16` | none | BF16 |
| Norms/HC/sinks | `BF16`/`F32` | none | BF16/F32 |

The advertised HF quantization config is:

```json
{
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "scale_fmt": "ue8m0",
  "weight_block_size": [128, 128]
}
```

That config is accurate for many dense linears, but it is not sufficient for
routed experts. `inference/config.json` separately states:

```text
dtype: fp8
expert_dtype: fp4
scale_fmt: ue8m0
```

## Existing AutoDeploy Capabilities

Relevant AutoDeploy components:

- `tensorrt_llm/_torch/auto_deploy/models/quant_config_reader.py`
  - Reads HF `quantization_config`.
  - Supports `quant_method` values including `fp8` and `mxfp4`.

- `tensorrt_llm/_torch/auto_deploy/transform/library/quantization.py`
  - `quantize_finegrained_fp8_linear_from_config`
  - Converts eligible linear ops to FineGrained FP8 quantized ops.

- `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/torch_quant.py`
  - `torch_fake_quant_finegrained_fp8_linear`
  - `trtllm_finegrained_fp8_linear`
  - Dynamic activation quantization plus block-scaled FP8 GEMM.

- `tensorrt_llm/_torch/auto_deploy/transform/library/quantize_moe.py`
  - FineGrained FP8 MoE transform.
  - NVFP4 MoE transform.
  - Does not directly model DeepSeek V4's FP8+packed-FP4 mixed checkpoint.

- `tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/mxfp4_moe.py`
  - `triton_mxfp4_moe`
  - `triton_mxfp4_moe_ep`
  - Triton-kernels-based MXFP4 MoE execution.

- `tensorrt_llm/_torch/auto_deploy/transform/library/mxfp4_moe.py`
  - Replaces router+dense MLP pattern with MXFP4 MoE.
  - Currently oriented toward `quant_method: mxfp4`.

- `tensorrt_llm/_torch/auto_deploy/transform/library/sharding_ir.py`
  - Has sharding support for `triton_mxfp4_moe`.
  - Rewrites to `triton_mxfp4_moe_ep`.

The useful pieces are present, but the automatic dispatch must be made
DeepSeek V4-aware.

## Key Gaps To Close

### 1. Generic `fp8` Handling Will Misclassify Routed Experts

If AutoDeploy only sees:

```json
"quant_method": "fp8"
```

then the natural generic path is FineGrained FP8 for all eligible linear/MoE
ops. That is wrong for routed experts because their weights are not FP8:

```text
layers.*.ffn.experts.*.w*.weight  I8
layers.*.ffn.experts.*.w*.scale   F8_E8M0
```

The support plan must override quantization per module family.

### 2. V4 Uses `.scale`, While Some AD Hooks Expect `weight_scale_inv`

AutoDeploy's FineGrained FP8 transforms currently expect or create buffers named
`weight_scale_inv`, while the V4 checkpoint stores sibling keys:

```text
layers.0.attn.wq_a.weight
layers.0.attn.wq_a.scale
```

The loader needs a V4-specific remap:

```text
<module>.weight + <module>.scale -> <module>.weight + <module>.weight_scale_inv
```

or the transform should accept `.scale` as an alias for FineGrained FP8.

### 3. E8M0 Scales Need Raw-Bit Awareness

E8M0 is exponent-only. In PyTorch, headers show `F8_E8M0`; locally this maps to
`torch.float8_e8m0fnu`.

For kernels that need FP32 scales, upcast by reinterpreting the exponent byte:

```text
exp_bits = scale.view(torch.uint8).to(torch.int32)
fp32_bits = exp_bits << 23
scale_fp32 = fp32_bits.view(torch.float32)
```

For kernels that expect raw exponent bytes, such as many MXFP4 layouts:

```text
good: loaded_scale.view(torch.uint8)
bad:  loaded_scale.to(torch.uint8)
```

Numeric conversion can destroy the exponent encoding.

### 4. Routed Expert Shapes Are Packed

Representative routed expert tensors:

```text
w1.weight  I8 [2048, 2048]  -> logical FP4 [2048, 4096]
w3.weight  I8 [2048, 2048]  -> logical FP4 [2048, 4096]
w2.weight  I8 [4096, 1024]  -> logical FP4 [4096, 2048]
```

The second dimension is halved because two FP4 values are packed per byte.

The scale shapes use 32-element FP4 blocks:

```text
w1.scale F8_E8M0 [2048, 128]  -> 4096 / 32 = 128
w3.scale F8_E8M0 [2048, 128]
w2.scale F8_E8M0 [4096, 64]   -> 2048 / 32 = 64
```

Any stacking, sharding, or conversion code must preserve this packed layout.

### 5. Current Modeling Fallback Is Not Production-Ready

The current `DeepseekV4MoE` scaffold can fall back to a dense expert loop when
`swiglu_limit > 0`. The real model has:

```text
swiglu_limit = 10.0
```

So production must not rely on the dense loop. It needs a V4-specific fused MoE
lowering that supports `swiglu_limit`.

## Proposed Quantization Architecture

Add a DeepSeek V4-specific quantization classification layer.

### Option A: Model-Specific Quant Config Reader

Introduce a reader/override that recognizes:

```text
model_type == "deepseek_v4"
quantization_config.quant_method == "fp8"
inference config expert_dtype == "fp4"
```

and returns an internal mixed quantization plan:

```yaml
quant_method: deepseek_v4_fp8
linear_quant_method: finegrained_fp8
expert_quant_method: mxfp4
scale_fmt: ue8m0
weight_block_size: [128, 128]
expert_block_size: 32
exclude_modules:
  - embed
  - head
  - "*.ffn.gate"
  - "*.attn.compressor"
  - "*.norm"
  - "*.hc_*"
  - "*.attn_sink"
  - "mtp.*"   # until MTP is supported
```

This is conceptually the same as vLLM's `DeepseekV4FP8Config`: standard FP8
behavior for non-MoE layers, MXFP4 behavior for `FusedMoE`.

### Option B: Transform-Level Detection

Keep the HF reader generic, but make quantization transforms inspect checkpoint
dtypes and module names:

- If module path matches `*.ffn.experts.*.w[123]` and checkpoint dtype is `I8`,
  route to MXFP4.
- If module path has paired `.scale` and checkpoint dtype is `F8_E4M3`, route to
  FineGrained FP8.
- If no matching quantized weight exists, keep BF16/F32.

This is more robust to future checkpoints but requires more loader metadata to
flow into transform decisions.

Recommended approach: start with Option A for clarity and speed, then factor
the useful pieces into transform-level detection once validated.

## Checkpoint Loading Plan

### Name Mapping

The custom model hierarchy should match checkpoint names where possible:

```text
embed.weight
layers.{i}.attn.wq_a.weight
layers.{i}.attn.wq_a.scale
layers.{i}.attn.wq_b.weight
layers.{i}.attn.wq_b.scale
layers.{i}.attn.wkv.weight
layers.{i}.attn.wkv.scale
layers.{i}.attn.wo_a.weight
layers.{i}.attn.wo_a.scale
layers.{i}.attn.wo_b.weight
layers.{i}.attn.wo_b.scale
layers.{i}.ffn.gate.weight
layers.{i}.ffn.gate.bias
layers.{i}.ffn.gate.tid2eid
layers.{i}.ffn.experts.{e}.w1.weight
layers.{i}.ffn.experts.{e}.w1.scale
layers.{i}.ffn.experts.{e}.w2.weight
layers.{i}.ffn.experts.{e}.w2.scale
layers.{i}.ffn.experts.{e}.w3.weight
layers.{i}.ffn.experts.{e}.w3.scale
layers.{i}.ffn.shared_experts.w1.weight
layers.{i}.ffn.shared_experts.w1.scale
...
```

For the first production path:

- Load normal transformer layers.
- Load `tid2eid` for the first three hash-routed layers.
- Skip `mtp.0.*`.
- Treat missing MTP keys as expected if the model omits MTP.
- Treat unexpected `mtp.0.*` keys as expected/skipped, not fatal.

### FineGrained FP8 Linears

For each eligible FP8 linear:

1. Keep `weight` as `torch.float8_e4m3fn`.
2. Load checkpoint `.scale`.
3. Store scale in an AD buffer accepted by the FineGrained FP8 op.
4. Upcast E8M0 to FP32 only when the backend requires FP32 scales.
5. Preserve E8M0 as raw bytes when the backend requires E8M0 bytes.

Eligible linears:

```text
layers.*.attn.wq_a
layers.*.attn.wq_b
layers.*.attn.wkv
layers.*.attn.wo_a
layers.*.attn.wo_b
layers.*.ffn.shared_experts.w1
layers.*.ffn.shared_experts.w2
layers.*.ffn.shared_experts.w3
```

Not initially eligible:

```text
layers.*.ffn.gate
layers.*.attn.compressor.*
embed
head
norms
hyper-connection tensors
attn_sink
mtp.*
```

### Routed MXFP4 Experts

For each layer's routed experts:

1. Load packed `I8` weights as raw bytes.
2. Load E8M0 scales as raw exponent bytes.
3. Stack expert weights into grouped MoE layout.
4. Concatenate gate/up path in the order expected by the chosen backend.
5. Keep down path separate.
6. Preserve scales with the same packing and expert order as weights.
7. Apply EP sharding only after the packed layout is established.

The desired logical layout is:

```text
gate_up_blocks:  [E_local, 2 * I, H / 32, 16]  uint8
gate_up_scales:  [E_local, 2 * I, H / 32]      uint8 or E8M0-backed
down_blocks:     [E_local, H, I / 32, 16]      uint8
down_scales:     [E_local, H, I / 32]          uint8 or E8M0-backed
```

The exact conversion depends on the backend:

- Triton-kernels MXFP4 path may swizzle via `convert_layout`.
- TRT-LLM/FlashInfer path may require NVFP4 block-scale interleave.
- DeepGEMM-style path may consume FP8 activations and FP4 weights directly.

The key invariant is that E8M0 scale bytes must remain exponent bytes until the
backend explicitly asks for FP32 scales.

## MoE Runtime Plan

DeepSeek V4 MoE consists of:

- BF16 router weight `[256, 4096]`
- optional router bias for non-hash layers
- `tid2eid` table for the first three hash-routed layers
- top-k = 6
- scoring function = `sqrtsoftplus`
- normalization of selected weights
- route scale = 1.5
- routed MXFP4 experts
- one shared FP8 expert
- `swiglu_limit = 10.0`

### Canonical Op Contract

Introduce:

```text
torch_deepseek_v4_moe(
    hidden_states,
    input_ids,
    router_weight,
    router_bias_or_none,
    tid2eid_or_none,
    routed_expert_packed_weights,
    routed_expert_scales,
    shared_expert_weights,
    shared_expert_scales,
    top_k,
    route_scale,
    swiglu_limit,
    is_hash_layer,
)
```

The source implementation may be a reference PyTorch path for export and unit
tests, but production lowering should replace it with fused kernels.

### Router Lowering

Hash-routed layers:

```text
indices = tid2eid[input_ids]
scores = sqrt(softplus(router_logits))
weights = gather(scores, indices)
weights = normalize(weights) * route_scale
```

Top-k layers:

```text
scores = sqrt(softplus(router_logits))
biased_scores = scores + bias
indices = topk(biased_scores, k=6)
weights = gather(scores, indices)
weights = normalize(weights) * route_scale
```

Add a fused `topk_sqrtsoftplus` kernel for non-hash layers. vLLM added an
analogous `topk_softplus_sqrt` path.

### Expert Lowering

Production path:

```text
router -> selected experts/weights
routed experts -> MXFP4 grouped MoE kernel
shared expert -> FineGrained FP8 SwiGLU or fused shared expert kernel
sum routed + shared
```

Important: the routed expert activation is not vanilla SwiGLU if
`swiglu_limit > 0`:

```text
up = clamp(up, -limit, limit)
gate = clamp(gate, max=limit)
hidden = silu(gate) * up
```

The selected MoE backend must support this activation variant, either directly
or via a small V4-specific fused activation wrapper.

## Attention Runtime Plan

DeepSeek V4 attention combines:

- Q low-rank path: `wq_a -> q_norm -> wq_b`
- per-head Q RMSNorm
- KV path: `wkv -> kv_norm`
- RoPE on the last 64 dims
- local sliding window of 128 tokens
- compressed KV memory for older context
- ratio-4 indexer top-k path
- ratio-128 compressed path
- attention sink
- inverse RoPE on output
- grouped output projection: `wo_a -> wo_b`

### Existing MLA Support Is Not Sufficient

AutoDeploy's existing classic MLA support is designed for DeepSeek V2/V3-style
MLA, not V4 HMA/MHC. It assumes different cache semantics and dimensions. V4
needs a new sparse attention descriptor and cache resource design.

### Canonical Op Contract

Introduce:

```text
torch_deepseek_v4_sparse_attention(
    hidden_states,
    position_ids,
    layer_id,
    compress_ratio,
    sliding_window,
    index_topk,
    q_weights,
    kv_weights,
    compressor_weights_or_none,
    indexer_weights_or_none,
    output_weights,
    attn_sink,
)
```

The exact source op can be split into smaller canonical ops if that makes
transform matching easier. The important point is that production lowering
should see a single semantic attention region rather than many opaque PyTorch
gather/einsum operations.

### Cache Resource Design

Add a DeepSeek V4 composite cache resource:

```text
swa_kv_cache
  local high-resolution KV for sliding window attention

mhc_cache
  compressed KV entries for ratio-4 and ratio-128 layers

indexer_cache
  compact indexer KV for ratio-4 top-k selection

compressor_state
  per-sequence partial-window state for decode when compression windows are
  incomplete
```

Recommended AutoDeploy runtime extension:

- Allow `SequenceInfo` and `CachedSequenceInterface` to carry named page tables
  per resource group.
- Keep all long-lived KV-like resources paged.
- Avoid unpaged `[max_batch, max_seq, ...]` V4 caches.

Reason: unpaged caches are not acceptable for 1M-context-capable models and can
disable important block-reuse behavior.

### Attention Kernel Roadmap

Bring-up:

1. BF16 correctness path for sparse attention.
2. Ratio-0/SWA layers first.
3. Ratio-128 compressed layers second.
4. Ratio-4 indexer layers third.

Production:

1. Fuse Q RMSNorm + RoPE.
2. Fuse KV RMSNorm + RoPE + cache insert.
3. Fuse compressor pooling + RMSNorm + RoPE + cache insert.
4. Quantize NoPE KV dims to FP8 E4M3 with E8M0 scales.
5. Keep RoPE dims BF16 unless a validated quantized path is added.
6. Add sparse attention over local window plus compressed/indexed entries.
7. Add fused inverse RoPE + output quantization before output projection.
8. Add optional FP4 indexer cache after FP8 path is stable.

vLLM's PR provides a useful reference structure:

- fused Q/K RMSNorm, RoPE, and KV insert
- fused compressor quant/cache insert
- fused indexer Q RoPE quant
- fused inverse RoPE FP8 quant
- sparse FlashMLA-style attention
- optional FP4 indexer cache on newer hardware

## CUDA Graph Plan

Use AutoDeploy's `torch-cudagraph` backend.

Recommended serving settings:

```yaml
compile_backend: torch-cudagraph

cuda_graph_config:
  batch_sizes: [1, 2, 4, 8, 16, 32, 64]

transforms:
  compile_model:
    piecewise_enabled: true
```

Requirements:

- Mark V4 sparse cached attention and V4 cache metadata prep as dynamic for
  piecewise capture.
- Capture static regions: RMSNorm, FP8 linears, shared expert, MoE post-router
  matmuls where shape buckets are fixed, output projections.
- Keep dynamic sparse/indexer/cache-update kernels graph-safe:
  - no allocation during replay
  - fixed output buffers
  - fixed metadata tensor shapes per bucket
  - deterministic workspace sizing
- Use decode batch padding through existing CUDA graph batch sizes.
- For mixed prefill/decode, rely on piecewise CUDA graph until the full dynamic
  V4 metadata path is graph-safe.

## Proposed AutoDeploy Config Direction

Initial correctness config:

```yaml
model_kwargs:
  skip_mtp: true
  ad_rope_cache_len: 8192

compile_backend: torch-cudagraph

runtime:
  enable_chunked_prefill: true

cuda_graph_config:
  batch_sizes: [1, 2, 4, 8, 16, 32, 64]

transforms:
  quantize_deepseek_v4_from_config:
    enabled: true
  compile_model:
    piecewise_enabled: true
  multi_stream_moe:
    enabled: false
```

Production config knobs to benchmark:

```yaml
max_batch_size: 32 | 64 | 128
max_seq_len: 8192 | 16384 | 32768
tokens_per_block: 32 | 64 | 128
expert_parallel_size: 8 | 16
attention_dp: false initially
multi_stream_moe: false initially, then benchmark
multi_stream_v4_attention: false initially, then benchmark
```

Do not target 1M context first. The advertised maximum is an architectural
capability, but production support should prove correctness and cache behavior
at shorter contexts before scaling.

## Implementation Milestones

### Milestone 1: Checkpoint Classifier

Deliverables:

- A read-only checkpoint scanner that categorizes every tensor.
- Counts by component and dtype.
- Assertions:
  - all routed expert weights are packed `I8`
  - all routed expert scales are `F8_E8M0`
  - all FP8 linears have paired `.scale`
  - compressor weights are BF16
  - `mtp.0.*` is either supported or explicitly skipped

Exit criteria:

- No unknown tensor families except explicitly waived keys.
- Classification matches the tensor counts from the safetensors index.

### Milestone 2: FineGrained FP8 Linear Loading

Deliverables:

- DeepSeek V4-specific `.scale` -> `weight_scale_inv` remap.
- E8M0 scale handling helper.
- FineGrained FP8 op support for attention and shared expert linears.

Validation:

- Pick one attention linear and one shared expert linear.
- Dequantize using a reference E8M0 helper.
- Compare AD op output against reference matmul.
- Test both unsharded and TP-sharded shapes.

### Milestone 3: Packed MXFP4 Expert Loader

Deliverables:

- Packed expert loader for `I8` weights.
- Raw-byte E8M0 scale preservation.
- Per-layer expert stacking.
- EP-compatible expert partitioning.

Validation:

- Unpack/dequantize one `w1`, `w2`, and `w3` expert against a reference.
- Compare one-token and multi-token expert outputs.
- Verify no numeric conversion is applied to E8M0 scale bytes.

### Milestone 4: V4 Router

Deliverables:

- Hash routing for first three layers using `tid2eid`.
- Fused or canonical `sqrtsoftplus` top-k router.
- Correct route normalization and scaling.

Validation:

- Compare hash-routed indices exactly.
- Compare top-k indices and weights against reference.
- Include bias path for non-hash layers.

### Milestone 5: Fused V4 MoE

Deliverables:

- `torch_deepseek_v4_moe` canonical op.
- Lowering to MXFP4 routed experts.
- Shared expert FineGrained FP8 path.
- Support for `swiglu_limit`.

Validation:

- Tiny-model MoE parity.
- Full hidden-size single-layer MoE parity on a small token batch.
- EP sharding parity across world sizes.
- CUDA graph replay test.

### Milestone 6: V4 Sparse Attention Source Op

Deliverables:

- Canonical sparse attention op or op region.
- Reference implementation for export tests.
- Existing modeling code converted away from opaque Python sparse loops where
  needed.

Validation:

- Ratio-0/SWA correctness.
- Ratio-128 compressed correctness.
- Ratio-4 indexer correctness.
- Chunked prefill equivalence.

### Milestone 7: Paged V4 Cache

Deliverables:

- Named page tables or composite V4 cache handler.
- SWA cache.
- MHC cache.
- Indexer cache.
- Per-sequence compressor state.

Validation:

- Decode equivalence vs full prefill.
- Chunked prefill equivalence.
- Prefix reuse and partial reuse tests.
- Long-context memory growth test.

### Milestone 8: Production Attention Kernels

Deliverables:

- Fused qnorm/rope.
- Fused kvnorm/rope/cache insert.
- Fused compressor/cache insert.
- Sparse attention kernel over local and compressed selected positions.
- Optional FP4 indexer cache path.

Validation:

- Kernel unit tests for numerics.
- End-to-end attention parity.
- CUDA graph replay.
- Nsight Systems/Compute sanity pass for launch count and occupancy.

### Milestone 9: End-to-End Serving

Deliverables:

- AutoDeploy registry YAML.
- `trtllm-serve` path.
- CUDAGraph-enabled decode path.
- Chunked prefill.

Validation:

- Reduced-layer run.
- Full 43-layer run.
- Deterministic prompt sanity tests.
- Throughput and latency benchmark.
- Regression test for checkpoint loading.

## Testing Matrix

### Unit Tests

- Checkpoint key classification.
- E8M0 upcast and raw-byte reinterpretation.
- FP8 linear reference.
- MXFP4 expert unpack/dequant reference.
- Router hash and top-k paths.
- `swiglu_limit` activation.
- Shared expert FP8 path.
- MTP skip behavior.

### Graph Tests

- Export with dynamic batch and sequence dimensions.
- Transform matching for V4 quantization.
- Transform matching for V4 MoE.
- Transform matching for V4 sparse attention.
- Shape propagation after each transform stage.

### Runtime Tests

- Single GPU reduced config.
- Multi-GPU TP.
- Expert parallel routed MoE.
- Chunked prefill.
- Decode with paged cache.
- CUDA graph decode replay.
- Piecewise CUDA graph mixed prefill/decode.

### Performance Tests

Metrics:

- TTFT by input length.
- TPOT by batch size.
- Tokens/sec under concurrency.
- GPU memory by context length.
- Kernel launch count.
- CUDA graph replay hit rate.
- Expert load balance.
- All-to-all overhead under EP.

Initial comparison target:

- vLLM's public DeepSeek V4 blog numbers on 16xH100 should be treated as an
  external sanity target, not a guaranteed requirement.

## Risk Register

| Risk | Impact | Mitigation |
| - | - | - |
| Generic FP8 quantization captures routed experts | Incorrect weights or load failure | Add V4-specific quant reader/transform |
| E8M0 scale numeric conversion | Severe accuracy loss | Use raw-byte view for MXFP4 scales |
| `.scale` naming mismatch | Missing scales or all-ones scales | Add V4 load hook/remap |
| Dense MoE fallback | Unusable performance | Make fused V4 MoE a release blocker |
| Existing MLA cache path mismatch | Incorrect decode or poor memory behavior | Add V4 sparse attention cache resource |
| Unpaged compressed caches | Memory blowup at long context | Use paged named cache resources |
| CUDA graph dynamic metadata | Replay failure or eager fallback | Piecewise capture and preallocated metadata |
| MTP checkpoint keys | Load warnings/failures | Explicitly skip until MTP support lands |
| Compressor BF16 cost | Prefill bottleneck | Optimize after correctness with fused compressor kernel |

## Open Questions

1. Which production MoE backend should be first?
   - Triton-kernels MXFP4 is closest to existing AD code.
   - TRT-LLM/FlashInfer MXFP4 may be faster if layout conversion is available.
   - DeepGEMM FP8xFP4 grouped GEMM may be best for Blackwell/Hopper depending
     on availability.

2. Should routed expert scales stay E8M0 bytes end-to-end?
   - Preferred for MXFP4 kernels.
   - FP32 upcast only for backends that explicitly require FP32 scales.

3. Should the FP8 `.scale` tensors be renamed to `weight_scale_inv` or should
   the FineGrained FP8 op accept `scale` directly?
   - Rename is least invasive.
   - Direct aliasing is cleaner for future HF checkpoints.

4. What is the first supported context length?
   - Recommendation: start with 8K or 16K.
   - Scale after paged V4 cache and chunked prefill are proven.

5. Should MTP be loaded later?
   - Initial serving can omit MTP.
   - MTP can be added after base model correctness and performance are stable.

## Recommended First Patch Series

1. Add DeepSeek V4 checkpoint classifier and tests.
2. Add DeepSeek V4 quant config override.
3. Add `.scale` -> `weight_scale_inv` load remap for FineGrained FP8.
4. Add E8M0 helpers:
   - upcast E8M0 to FP32
   - reinterpret E8M0 as raw `uint8`
5. Add packed MXFP4 expert loader and stacking.
6. Add V4 router canonical op or fused router transform.
7. Add `torch_deepseek_v4_moe` and lower to MXFP4 routed experts.
8. Add V4 sparse attention canonical op.
9. Add V4 paged cache resource handler.
10. Add CUDA graph dynamic-op registration and piecewise capture support.

This order keeps correctness and weight loading ahead of kernel performance
work, while still setting up the final high-performance serving path.
