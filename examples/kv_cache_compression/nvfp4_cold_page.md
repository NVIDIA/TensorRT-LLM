<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVFP4 Cold-Page KV-Cache Compression

NVFP4 cold-page compression stores supported Attention KV buffers in NVFP4
while their Pages reside in a Host or Disk cache tier. It is a storage-boundary
optimization: the active GPU cache and Attention kernels continue to use the
model's normal runtime KV type.

## Motivation

Agentic workloads repeatedly revisit long prompts, tool histories, and
intermediate reasoning states. Their reusable KV cache can therefore grow
faster than the GPU, Host, or Disk cache capacity available to the serving
system. When the secondary cache cannot retain enough reusable Pages, useful
prefixes are evicted, the cache hit rate falls, and TensorRT-LLM must recompute
more context. Under sustained pressure, this can sharply increase time to first
token (TTFT) and reduce throughput.

NVFP4 cold-page compression addresses both sides of this bottleneck. It stores
more reusable Pages within the same Host or Disk byte quota, which can improve
the cache hit rate when cold-tier capacity is the limiting factor. It also
reduces the bytes moved during GPU-to-Host, Host-to-GPU, Host-to-Disk, and
Disk-to-Host migration. The benefit is workload-dependent: workloads with
substantial prefix reuse and real cold-tier pressure have the most opportunity
to reduce recomputation and migration cost.

| Source of pressure | What compression changes | Expected effect |
| --- | --- | --- |
| Host or Disk capacity | Supported Attention KV occupies fewer cold-tier bytes | More reusable Pages can remain cached and the hit rate can increase |
| Page migration | Fewer bytes cross the storage boundary | Offload and onboard can complete faster |
| GPU runtime compatibility | The hot Page is restored to FP16, BF16, or FP8 before publication | Attention and the model runtime do not consume the cold representation |

## Design and Enablement

Enable the feature through the normal LLM configuration. `KvCacheConfig`
selects the native C++ KVCacheManagerV2 and provisions the cold tiers;
`ColdPageQuantizationCompressionConfig` selects the NVFP4 representation used
at those tiers.

```text
LLM / trtllm-serve configuration
  |
  +-- KvCacheConfig
  |     +-- use_kv_cache_manager_v2 = true
  |     +-- runtime KV dtype = auto / FP16 / BF16 / FP8
  |     `-- Host cache and/or Disk cache is provisioned
  |
  `-- ColdPageQuantizationCompressionConfig(quant="nvfp4")
                       |
                       v
Cold-page compression manager
  +-- selects compressible Attention buffer roles
  +-- derives compact offsets and optional K/V global scales
  `-- creates and registers the NVFP4 cold-page codec
                       |
                       v
C++ KVCacheManagerV2
  +-- owns Pages, Slots, cache levels, migration batches, streams, and events
  +-- hot -> cold: fused NVFP4 encode + transfer
  `-- cold -> hot: fused transfer + decode to the runtime KV type
                       |
                       v
KVCM publishes the destination Page after completion
  `-- Attention consumes only the normal GPU hot-page layout
```

KVCacheManagerV2 remains the storage and lifecycle owner. The compression
manager owns the representation policy: buffer-role selection, compact layout,
scales, and codec dispatch. This separation keeps Page identity, scheduling,
reuse policy, allocation, and the Attention-visible GPU layout unchanged.

The Page data path is:

```text
GPU hot Page (FP16/BF16/FP8 Attention KV)
    |  KVCM selects a migration batch
    |  fused NVFP4 encode + GPU-to-Host transfer
    v
Host cold Page (NVFP4 Attention data + block scales;
                non-Attention and auxiliary buffers remain lossless)
    |  optional pinned-Host staging to or from Disk
    |  fused Host-to-GPU transfer + decode
    v
GPU hot Page (original runtime KV type)
    |  completion event
    v
KVCM publishes the new cache level and Slot mapping
```

## Feature Behavior

Each cold Page is one compact storage blob. For conventional MHA/GQA, the K and
V buffers are encoded as packed NVFP4 data plus one E4M3 block scale per 16
scalar values. For key-only MLA, the latent Attention key is encoded. Auxiliary
roles in the same Attention lifecycle, such as a DSA index key, are appended to
the blob losslessly. Non-Attention lifecycles, including GDN, SSM, and Conv
state, use the default lossless cold-page codec and are not quantized.

Disk migration uses pinned Host staging. TensorRT-LLM first produces the same
compact cold Page used by the Host tier, writes that representation to Disk,
and reads it back into staging before decode. Host and Disk therefore share one
cold representation rather than introducing a separate Disk quantization
format.

By default, the codec uses identity K/V global scales, so a regular checkpoint
requires no calibration step. Dynamic E4M3 block scales are still computed for
every group of 16 values. A compatible ModelOpt NVFP4 checkpoint can optionally
supply per-layer K/V global-scale metadata; see
[Optional ModelOpt Global Scales](#optional-modelopt-global-scales).

This is different from an active NVFP4 KV cache. Do not set
`kv_cache_config.dtype: nvfp4` for this feature. Cold-page compression does not
change Page identity, token count, block reuse, scheduling, or the
Attention-visible GPU layout.

## Support Matrix

| Dimension | Status | Behavior |
| --- | --- | --- |
| Backend and hardware | Supported | PyTorch backend, native C++ KVCacheManagerV2, and SM100 or SM103 GPU |
| Hot Attention KV dtype | Supported | FP16, BF16, or FP8; the Page is decoded before Attention reads it |
| MHA and GQA | Supported | Conventional K and V buffers are stored as NVFP4 data plus block scales |
| Key-only MLA | Supported | The latent Attention key is compressed; auxiliary buffers remain lossless |
| Host and Disk cold tiers | Supported | Both tiers use the same compact representation; Disk uses pinned-Host staging |
| KV-cache block reuse | Supported | Page and token identity are unchanged |
| GDN, SSM, and Conv state | Not quantized | Non-Attention lifecycles preserve their original representation through the lossless codec |
| DSA and other side buffers | Not quantized | `index_key` and other auxiliary roles are copied losslessly beside compressed Attention data |
| Speculative decoding | Limited | One-model MTP-EAGLE and one-model EAGLE3 |
| HELIX context parallelism | Not supported | Rejected during configuration |
| DeepSeek-V4 specialized cache | Not supported | Its already-compressed cache uses dedicated `deepseek_v4_*` roles instead of the standard K/V or key-only MLA contract, so configuration fails closed |

Set `kv_cache_config.use_kv_cache_manager_v2: true` explicitly, and do not set
`TLLM_KV_CACHE_MANAGER_V2_BACKEND=python`. A nonzero Host or Disk cache is also
required for Pages to cross a compression boundary.

On Linux 6.11 through 6.13, mixed models that need both NVFP4 Attention
lifecycles and lossless SSM/GDN fallback lifecycles are currently rejected.
Those kernels require chunked pinned-memory registration, which the embedded
lossless fallback cannot yet split safely at registration boundaries.

### Model Compatibility and Validation

The following model families cover the major cache structures exercised during
development. Exact checkpoint, parallelism, Attention-backend, and hardware
requirements still apply.

| Model family | Attention and cache structure | Validation status | Cold-page behavior |
| --- | --- | --- | --- |
| Qwen3 | Conventional MHA/GQA K/V | Validated | K and V are compressed to NVFP4 |
| Qwen3.5 | GQA Attention plus GDN/SSM state | Validated | Attention K/V is compressed; recurrent state remains lossless |
| GLM-5.2 (GLM family) | Key-only MLA plus DSA side data | Validated | The latent key is compressed; DSA index and side buffers remain lossless |
| DeepSeek-R1/V3 family | Key-only MLA | Supported by the MLA contract; the DeepSeek-V3.2 layout is directly validated | The latent Attention cache is compressed; auxiliary data remains lossless |
| DeepSeek-V4 | Specialized `deepseek_v4_*` cache roles | Not currently supported | The already-compressed cache does not enter the generic NVFP4 cold-page codec |

## Qwen3.5-4B on One GPU

The following example uses a regular BF16 checkpoint. Attention KV stays BF16
on the GPU and uses NVFP4 only while it resides in the 8 GiB Host tier.

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (
    ColdPageQuantizationCompressionConfig,
    KvCacheConfig,
)

with LLM(
    model="Qwen/Qwen3.5-4B",
    backend="pytorch",
    trust_remote_code=True,
    max_seq_len=4096,
    max_batch_size=4,
    kv_cache_config=KvCacheConfig(
        use_kv_cache_manager_v2=True,
        dtype="auto",
        enable_block_reuse=False,
        host_cache_size=8 << 30,
    ),
    kv_cache_compression_config=ColdPageQuantizationCompressionConfig(
        quant="nvfp4",
    ),
) as llm:
    outputs = llm.generate(
        ["Explain why prefix caching helps agentic workloads."],
        SamplingParams(max_tokens=128, temperature=0.0),
    )
    print(outputs[0].outputs[0].text)
```

The equivalent `trtllm-serve` configuration is:

```yaml
backend: pytorch
trust_remote_code: true

kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: auto
  enable_block_reuse: false
  host_cache_size: 8589934592  # 8 GiB

kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: nvfp4
```

```bash
trtllm-serve Qwen/Qwen3.5-4B --config qwen3.5-cold-nvfp4.yaml
```

This minimal smoke disables block reuse to keep the example small. For an
agentic prefix-reuse workload, enable block reuse and supply enough reusable
long-context traffic to create real GPU KV pressure. Compression preserves the
Page identity used by block reuse.

## GLM-5.2 Disaggregated Serving

This example uses the NVIDIA NVFP4 weight checkpoint on one 8-GPU SM100/SM103
node, split into a 4-GPU context worker and a 4-GPU generation worker. Weight
quantization and cold-page compression are independent: active KV is still FP8,
and supported cold Attention buffers use NVFP4.

Cold-page compression and disaggregated transfer cover different boundaries:

```text
worker-local GPU <-> Host/Disk    cold-page codec (NVFP4)
context GPU -> generation GPU     NIXL (active/hot KV representation)
```

Create `context.yaml`:

```yaml
enable_attention_dp: true
enable_chunked_prefill: true
disable_overlap_scheduler: true
stream_interval: 10
trust_remote_code: true

kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.7
  host_cache_size: 17179869184  # 16 GiB per rank

kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: nvfp4

cuda_graph_config: null
moe_config:
  backend: CUTEDSL
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
```

Create `generation.yaml`:

```yaml
enable_attention_dp: true
enable_chunked_prefill: true
stream_interval: 10
trust_remote_code: true

kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.7
  host_cache_size: 17179869184  # 16 GiB per rank

kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: nvfp4

cuda_graph_config:
  enable_padding: true
  max_batch_size: 128
moe_config:
  backend: CUTEDSL
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
```

Start the context and generation workers in separate shells on one 8-GPU node:

```bash
# Context worker
CUDA_VISIBLE_DEVICES=0,1,2,3 trtllm-serve nvidia/GLM-5.2-NVFP4 \
  --backend pytorch --served_model_name nvidia/GLM-5.2-NVFP4 \
  --host 0.0.0.0 --port 8001 \
  --max_batch_size 128 --max_num_tokens 16640 --max_seq_len 8232 \
  --tp_size 4 --ep_size 4 --pp_size 1 \
  --config context.yaml

# Generation worker
CUDA_VISIBLE_DEVICES=4,5,6,7 trtllm-serve nvidia/GLM-5.2-NVFP4 \
  --backend pytorch --served_model_name nvidia/GLM-5.2-NVFP4 \
  --host 0.0.0.0 --port 8002 \
  --max_batch_size 128 --max_num_tokens 10240 --max_seq_len 10240 \
  --tp_size 4 --ep_size 4 --pp_size 1 \
  --config generation.yaml
```

Create `disaggregated.yaml` for the two workers on the same node:

```yaml
hostname: 0.0.0.0
port: 8000
backend: pytorch
context_servers:
  num_instances: 1
  urls:
    - "localhost:8001"
generation_servers:
  num_instances: 1
  urls:
    - "localhost:8002"
```

Start the orchestrator and send a request:

```bash
trtllm-serve disaggregated -c disaggregated.yaml

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/GLM-5.2-NVFP4",
    "prompt": "Explain disaggregated LLM serving in three sentences.",
    "max_tokens": 64,
    "temperature": 0
  }'
```

The orchestrator does not own a KV cache. Put `kv_cache_config` and
`kv_cache_compression_config` on each worker that uses a local Host or Disk
tier. A successful NIXL context-to-generation transfer alone does not prove
that the worker migrated a Page to its cold tier. This configuration is a
functional disaggregated-serving example, not an Agentic block-reuse
benchmark; enable block reuse and use a reuse-bearing workload when measuring
the capacity and hit-rate benefit.

## Verify Activation

Configuration alone does not prove that a Page crossed the compression
boundary. A short request may remain entirely on the GPU. Run enough concurrent
or reusable long-context requests to create GPU KV pressure, then inspect the
server's `/metrics` endpoint. Both of the following counters should become
nonzero when the workload offloads and later reuses cold Pages:

* `trtllm_kv_cache_offload_bytes_total`
* `trtllm_kv_cache_onboard_bytes_total`

Also verify the resolved worker configuration. In disaggregated serving, each
context or generation worker owns its own local KVCM and cold-tier quota; the
front-end orchestrator does not.

## Optional ModelOpt Global Scales

By default, the cold-page codec uses identity K/V global scales, so a regular
checkpoint requires no calibration step. If a local ModelOpt NVFP4 checkpoint
for the same model contains per-layer K/V scale metadata, pass its directory:

```yaml
kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: nvfp4
  scale_checkpoint_path: /path/to/modelopt-nvfp4-checkpoint
```

`scale_checkpoint_path` supplies optional metadata; it is not the model path.
TensorRT-LLM does not derive KV activation scales from ordinary model weights.
These per-layer K/V global scales apply to the conventional two-buffer K/V
layout. Key-only MLA and draft-model cold Pages currently use identity global
scales.

## Enablement Checklist

1. Run the PyTorch backend on an SM100 or SM103 GPU.
2. Select the native C++ KVCacheManagerV2 with
   `use_kv_cache_manager_v2: true`.
3. Keep `kv_cache_config.dtype` at the model's intended runtime KV type:
   `auto`, FP16/BF16, or FP8.
4. Provision a nonzero Host cache and, optionally, a Disk cache. A positive
   `disk_cache_size` also requires `disk_cache_path` to name an existing
   directory.
5. Set `kv_cache_compression_config.algorithm` to
   `quantization_for_cold_page` and `quant` to `nvfp4`.
6. Apply both cache configurations to every worker that owns a local cold tier.
7. Verify nonzero offload and onboard bytes under a workload that creates real
   KV pressure.

In this design, `KvCacheConfig` and KVCacheManagerV2 own capacity, Page/Slot
allocation, cache-level routing, migration batching, completion events, and
mapping publication. `ColdPageQuantizationCompressionConfig` and its codec own
only the cold representation and its encode/decode policy. This is why the
feature can reduce the cold-tier storage footprint and transfer pressure
without changing the model's Attention path.

For the underlying KVCM lifecycle and codec ABI, see
[KVCacheManagerV2 Cold-Page Codec Design](../../docs/source/developer-guide/kv-cache-cold-page-codec.md).
