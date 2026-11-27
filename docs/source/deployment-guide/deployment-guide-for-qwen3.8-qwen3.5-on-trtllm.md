<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment Guide for Qwen3.8 MoE and Qwen3.5 MoE on TensorRT LLM - Blackwell Hardware

## Introduction

This guide describes how to serve the Qwen3.8 MoE and Qwen3.5 MoE hybrid models with the TensorRT LLM PyTorch backend. It covers the following checkpoints:

* **Qwen3.8-2.4T-A95B MoE FP8** — 2.4 trillion total parameters and 95 billion active parameters per token.
* **Qwen3.8-27B dense NVFP4** — 27 billion parameters with mixed NVFP4 and FP8 quantization.
* **Qwen3.5-397B-A17B MoE NVFP4** — 397 billion total parameters and 17 billion active parameters per token.

The models share the `qwen3_5_moe_text` decoder architecture and use the same TensorRT LLM implementation, registered as `Qwen3_5MoeForCausalLM`. Both interleave three gated-delta-network (GDN) linear-attention layers with one grouped-query-attention (GQA) layer, and both use 512 routed experts with top-10 routing. The dense Qwen3.8-27B checkpoint uses the `Qwen3_5ForConditionalGeneration` multimodal wrapper and the `Qwen3_5ForCausalLM` text decoder.

Sharing an implementation does not make their deployment configurations interchangeable. Qwen3.8 MoE is substantially larger, uses an FP8 checkpoint, and is text-only. The NVIDIA Qwen3.5 MoE NVFP4 checkpoint uses a multimodal wrapper around the shared text decoder. Parallelism, quantization backends, cache sizing, and expert placement must be selected for the exact checkpoint.

## Architecture and Checkpoint Differences

| Property | Qwen3.8-2.4T-A95B MoE FP8 | Qwen3.5-397B-A17B MoE NVFP4 |
|---|---:|---:|
| Hugging Face text architecture | `Qwen3_5MoeForCausalLM` | `Qwen3_5MoeForCausalLM` inside `Qwen3_5MoeForConditionalGeneration` |
| Decoder layers | 92 | 60 |
| Hidden size | 8192 | 4096 |
| GDN / GQA layers | 69 / 23 | 45 / 15 |
| Routed experts / experts per token | 512 / 10 | 512 / 10 |
| Built-in MTP modules | 1 recurrent module | 1 recurrent module |
| Recommended checkpoint precision in this guide | Block-scaled FP8 routed experts | NVFP4 with FP8 KV cache |
| Modality | Text | Text, image, and video |

The Qwen3.8 MoE FP8 checkpoint quantizes the routed expert projections to FP8 (E4M3) with 128x128 weight block scales and dynamic activation scaling. Attention, GDN projections, shared experts, routers, embeddings, and the LM head are excluded from quantization and retain their checkpoint dtypes.

The serving configuration controls the attention KV-cache dtype and GDN recurrent-state dtype independently through `kv_cache_config.dtype` and `kv_cache_config.mamba_ssm_cache_dtype`. The GDN state supports `auto`, `float16`, `bfloat16`, and `float32`.

## Support Status

| Capability | Qwen3.8 MoE FP8 | Qwen3.5 MoE NVFP4 |
|---|---|---|
| Aggregated serving | Supported | Supported |
| Disaggregated serving | Supported | Supported |
| MTP speculative decoding | Supported | Supported |
| Static EPLB | Supported | Supported |
| Online EPLB | Supported | Supported |
| Attention data parallelism and expert parallelism | Supported | Supported |
| GDN state replay | Supported | Supported |
| KV cache manager V2 | Supported and default | Supported and default |

## Performance

The performance reference uses GB300, aggregated serving, and an exact 8192-input/1024-output-token workload. Low-latency profiles use TP16/EP1 on 16 GPUs at concurrency 1. High-throughput profiles use attention DP with TP32/EP32 on 32 GPUs and a 544-slot static EPLB map. Low-latency throughput is output tokens per second per user (`1000 / median TPOT`); high-throughput is total input-plus-output tokens per second per GPU.

The following are the best audited Qwen3.8 MoE FP8 checkpoint points across the validated implementation variants:

| Objective | Speculative decoding | Topology | Concurrency | Best audited value |
|---|---|---|---:|---:|
| Low latency | Disabled | TP16/EP1 | 1 | **133.502 output tok/s/user** (7.491 ms median TPOT) |
| Low latency | MTP3 | TP16/EP1 | 1 | **383.051 output tok/s/user** (2.611 ms median TPOT) |
| High throughput | Disabled | Attention DP32/EP32, Static544 | 3264 | **4059.200 total tok/s/GPU** |
| High throughput | MTP3 | Attention DP32/EP32, Static544 | 2304 | **4118.164 total tok/s/GPU** |

MTP3 performance results use a controlled accepted-draft count of 2.3.

## Prerequisites

* GPU: NVIDIA Blackwell or Hopper Architecture
* OS: Linux
* Drivers: CUDA Driver 575 or Later
* Docker with NVIDIA Container Toolkit installed
* Python3 and python3-pip (Optional, for accuracy evaluation only)

## Models

* [Qwen/Qwen3.8-2.4T-A95B-FP8](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B-FP8) (FP8; the checkpoint the Qwen3.8 profiles in this guide target)
* [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) (base, BF16)
* [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
* [Inferact/Qwen3.8-27B-NVFP4](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4)
* [nvidia/Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)
* [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) (base, BF16)

## GPU Requirements

The following table lists the minimum validated GPU counts for the checkpoints and platforms covered by this guide:

| Checkpoint | Platform | Minimum GPUs |
|---|---|---|
| Qwen3.8-2.4T-A95B MoE FP8 | B200 | 16x B200 |
| Qwen3.8-2.4T-A95B MoE FP8 | B300 | 16x B300 |
| Qwen3.8-2.4T-A95B MoE FP8 | GB200 | 16x GB200 |
| Qwen3.8-2.4T-A95B MoE FP8 | GB300 | 16x GB300 |
| Qwen3.5-397B-A17B MoE NVFP4 | B200 | 4x B200 |
| Qwen3.5-397B-A17B MoE NVFP4 | B300 | 4x B300 |
| Qwen3.5-397B-A17B MoE NVFP4 | GB200 | 4x GB200 |
| Qwen3.5-397B-A17B MoE NVFP4 | GB300 | 4x GB300 |

The Qwen3.8 MoE FP8 checkpoint uses `tensor_parallel_size: 16` for the minimum-GPU configurations on B200, B300, GB200, and GB300. The Qwen3.5 MoE NVFP4 checkpoint has been validated with `tensor_parallel_size: 4`.

## Deployment Steps

### Run Docker Container

Run the docker container using the TensorRT LLM NVIDIA NGC image.

```shell
docker run --rm -it \
--ipc=host \
--gpus all \
-p 8000:8000 \
-v ~/.cache:/root/.cache:rw \
--name tensorrt_llm \
nvcr.io/nvidia/tensorrt-llm/release:x.y.z \
/bin/bash
```

Note:

* The command mounts your user `.cache` directory to save the downloaded model checkpoints which are saved to `~/.cache/huggingface/hub/` by default. This prevents having to redownload the weights each time you rerun the container. If the `~/.cache` directory doesn't exist please create it using `$ mkdir ~/.cache`.
* You can mount additional directories and paths using the `-v <host_path>:<container_path>` flag if needed, such as mounting the downloaded weight paths.
* The command also maps port `8000` from the container to your host so you can access the LLM API endpoint from your host.
* See the <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags> for all the available containers. The containers published in the main branch weekly have `rcN` suffix, while the monthly release with QA tests has no `rcN` suffix. Use the `rc` release to get the latest model and feature support.

If you want to use latest main branch, you can choose to build from source to install TensorRT LLM, the steps refer to [https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html](https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html).

### Recommended Performance Settings

We maintain YAML configuration files with recommended performance settings in the [`examples/configs`](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/configs) directory. These config files are present in the TensorRT LLM container at the path `/app/tensorrt_llm/examples/configs`. You can use these out-of-the-box, or adjust them to your specific use case.

Set the TensorRT LLM directory and select one of the configuration files listed below:

```shell
TRTLLM_DIR=/app/tensorrt_llm # change as needed to match your environment
cd "${TRTLLM_DIR}"
EXTRA_LLM_API_FILE=${TRTLLM_DIR}/examples/configs/curated/qwen3.8-low-latency.yaml
```

| Checkpoint | Profile | Configuration file |
|---|---|---|
| Qwen3.8-2.4T-A95B MoE FP8 | Low latency | [`qwen3.8-low-latency.yaml`](../../../examples/configs/curated/qwen3.8-low-latency.yaml) |
| Qwen3.8-2.4T-A95B MoE FP8 | Low latency with MTP3 | [`qwen3.8-low-latency-mtp3.yaml`](../../../examples/configs/curated/qwen3.8-low-latency-mtp3.yaml) |
| Qwen3.8-2.4T-A95B MoE FP8 | High throughput | [`qwen3.8-high-throughput.yaml`](../../../examples/configs/curated/qwen3.8-high-throughput.yaml) |
| Qwen3.8-2.4T-A95B MoE FP8 | High throughput with MTP3 | [`qwen3.8-high-throughput-mtp3.yaml`](../../../examples/configs/curated/qwen3.8-high-throughput-mtp3.yaml) |
| Qwen3.5-397B-A17B MoE NVFP4 | General deployment | [`qwen3.5.yaml`](../../../examples/configs/curated/qwen3.5.yaml) |

The MTP draft length is configurable through `speculative_config.max_draft_len`; the MTP3 profiles set it to 3. The high-throughput profiles use the following static expert placements:

| Profile | Static EPLB configuration |
|---|---|
| High throughput without MTP | [`qwen3.8-ep32-static544.yaml`](../../../examples/configs/curated/eplb/qwen3.8-ep32-static544.yaml) |
| High throughput with MTP3 | [`qwen3.8-ep32-static544-mtp3.yaml`](../../../examples/configs/curated/eplb/qwen3.8-ep32-static544-mtp3.yaml) |

Each file assigns expert IDs to 544 global slots for every MoE layer. Repeated expert IDs create replicas of heavily routed experts. `layer_updates_per_iter: 0` keeps the assignment static. The no-MTP map covers the 92 target-model layers, while the MTP3 map also includes the MTP layer. These placements are specific to Qwen3.8-2.4T-A95B, EP32, the selected MTP mode, and the benchmark traffic distribution. Generate a new placement when any of those inputs changes. See the [EPLB example](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/wide_ep/ep_load_balancer) for collecting routing statistics and generating a placement file.

### Experimental QKV Preprocessing Tiling

`TRTLLM_ENABLE_QKV_ROPE_TILING=1` enables experimental eight-token tiling for fused Q/K normalization, rotary embedding, and gate extraction. It is disabled by default. Set the environment variable before starting `trtllm-serve` or importing TensorRT LLM in Python.

Tiling applies to preprocessing batches with at least 256 tokens, at least 3,072 token/head pairs across Q/K/V (using local head counts after tensor parallelism), and `(head dimension, rotary dimension)` of `(128, 128)` or `(256, 64)`. Other workloads use the single-token path. Performance has been measured on B200 and Rubin; other GPU architectures can opt in, but their performance has not been established by those measurements.

Tiling can change Q/K rounding. Changing batch size across a dispatch boundary can therefore change cached K values and generated tokens. Leave the flag unset when batch-invariant results are required.

### Disabling Eager Fusions

`TRTLLM_QWEN3_EAGER_FUSION_DISABLED=1` turns off the fused all-reduce and RMSNorm kernels and the FP8 copies that RMSNorm writes for static FP8 layers, which can help when bisecting a numerics issue.

### Launch the TensorRT LLM Server

Below is an example command to launch the TensorRT LLM server from within the container with the checkpoint that matches the configuration file you selected above.

```shell
trtllm-serve <model_path_or_hf_id> --host 0.0.0.0 --port 8000 --reasoning_parser qwen3_5 --tool_parser qwen3 --config ${EXTRA_LLM_API_FILE}
```

Qwen3.8 and Qwen3.5 both use the `qwen3_5` reasoning parser (their chat template pre-injects a `<think>` block, so reasoning starts at the beginning of the response). The `qwen3` tool parser handles the Qwen3 function-call format.

After the server is set up, the client can now send prompt requests to the server and receive results.

## Qwen3.5 GB300 NVL72 Agentic Disaggregated Serving

Use this profile for long, multi-turn **text** conversations with
`nvidia/Qwen3.5-397B-A17B-NVFP4-V2` on GB300 NVL72. It separates prefill and
decode, enables MTP with six draft tokens on decode, and offloads the KV cache
to host memory. The earlier Qwen3.5 general-deployment profile and the Qwen3.8
profiles above target different workloads and checkpoints.

For the newer TensorRT LLM 1.4 and Dynamo 1.8 profiles with the remote KV cache
level, see
[Qwen3.5 GB300 NVL72 AgentX Disaggregated Serving with Remote KV Cache](#qwen35-gb300-nvl72-agentx-disaggregated-serving-with-remote-kv-cache).

The serving settings come from [InferenceX #4195](https://github.com/SemiAnalysisAI/InferenceX/pull/4195)
at commit `76fde8ec76a8573c350b553f7c2583088bf6558f`. Its
[sweep](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39016296529)
passed nine performance jobs and six applicable eval jobs. The performance
jobs use synthetic speculative acceptance; these production commands leave
acceptance to the model. Those throughput results do not establish production
throughput with natural acceptance, or validate these Docker/MPI launch commands.

### Runtime and Memory Requirements

Use the exact versions together:

| Component | Version |
|---|---|
| TensorRT-LLM container | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0a0.dev202610151219` |
| `ai-dynamo` and `ai-dynamo-runtime` wheels | `1.6.0.dev20261010` |
| Checkpoint | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` |

This is a development-container profile. Requalify the checkpoint, configuration,
and transport together when upgrading. The configuration files added to this
guide must come from your TensorRT-LLM checkout; they are not bundled in the older
container listed above.

The starting 1P1D profile uses three GPUs on one four-GPU GB300 node: one prefill
GPU and two decode GPUs. Each rank reserves **128 GiB of host KV cache**
(`host_cache_size: 137438953472`), requiring 384 GiB for these caches on that
node, plus memory for weights, loading, staging, and the operating system. The
source system has 840 GiB of host memory per four-GPU node. A fully occupied
four-rank node reserves 512 GiB for host KV cache. Check the remaining host-memory
budget before increasing worker count or cache size.

These profiles do not use the
[remote KV cache level](../features/kvcache.md#remote-cache-level-in-donated-host-memory-prototype).
If you add it, every rank with `remote_cache_config.donor.size` also holds that
many bytes of host memory for the life of its process, on top of its
`host_cache_size`. For example, decode ranks that donate 100 GiB each on a full
four-rank node with 128 GiB host caches use 912 GiB, which exceeds the 840 GiB of
the source system. Lower `host_cache_size` or `donor.size` on the donating ranks
so that host cache plus donation fits each node's budget.

Use a GB300 NVL72 system with working GPU peer access and NIXL/UCX transport.
For workers spanning nodes, the hosts must share the required NVLink fabric and
have compatible MPI, drivers, networking, and container mounts. Have reachable
etcd and NATS services before starting Dynamo; see the
[Dynamo CLI installation guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/cli/installation/install-dynamo.mdx).

### Prepare the Container

Download the exact V2 checkpoint into a local directory. Run the following from
the TensorRT-LLM checkout, replacing `MODEL_DIR` with that directory:

```bash
export MODEL_DIR=/absolute/path/to/Qwen3.5-397B-A17B-NVFP4-V2
docker run --rm -it --name qwen35-gb300 \
    --gpus all --network host --ipc host --ulimit memlock=-1 \
    --mount type=bind,src="$MODEL_DIR",dst=/models/qwen35,readonly \
    --mount type=bind,src="$PWD",dst=/opt/trtllm-recipes,readonly \
    nvcr.io/nvidia/tensorrt-llm/release:1.3.0a0.dev202610151219 \
    /bin/bash
```

Inside the container, install the matching public wheels:

```bash
python3 -m pip install --pre \
    --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com \
    ai-dynamo==1.6.0.dev20261010 ai-dynamo-runtime==1.6.0.dev20261010
python3 -c 'import importlib.metadata as m; print({p: m.version(p) for p in ("tensorrt-llm", "ai-dynamo", "ai-dynamo-runtime")})'
```

Open a shell in this container for each of the frontend, prefill, and decode
commands below with `docker exec -it qwen35-gb300 /bin/bash`. Run this setup in
**each shell**. Replace the service endpoints when they run on another host:

```bash
export MODEL_PATH=/models/qwen35
export CONFIG_DIR=/opt/trtllm-recipes/examples/configs/curated
export ETCD_ENDPOINTS=http://127.0.0.1:2379
export NATS_SERVER=nats://127.0.0.1:4222
export DYN_NAMESPACE=qwen35-gb300
export DYN_DISCOVERY_BACKEND=etcd
export DYN_REQUEST_PLANE=tcp
source "$CONFIG_DIR/qwen3.5-gb300-agentic.env"
unset TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS
```

The [worker environment](../../../examples/configs/curated/qwen3.5-gb300-agentic.env)
preserves the source profile's allocator, CUDA launch, GC, weight-staging, transfer,
and conversation-affinity settings. It includes an intentionally empty
`MIMALLOC_PURGE_DELAY`. Keep the draft head and its precision as shipped with the
V2 checkpoint. Never set `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` in production
or accuracy runs.

### Start the Frontend and Workers

In the frontend shell:

```bash
export ETCD_LEASE_TTL=120
export DYN_TOKENIZER_CACHE=1
export DYN_TOKENIZER_CACHE_BYTES=8000000000
export DYN_TOKENIZER=fastokens
python3 -m dynamo.frontend --http-port 8000 \
    --router-mode kv --router-session-affinity-ttl-secs 14400
```

In the prefill shell:

```bash
export CUDA_VISIBLE_DEVICES=0
export DYN_SYSTEM_PORT=8081
export CUDA_SCALE_LAUNCH_QUEUES=4x
export TRTLLM_SKIP_KV_CACHE_ESTIMATION=1
export TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1
mpirun --allow-run-as-root -np 1 --bind-to none \
    trtllm-llmapi-launch python3 -m dynamo.trtllm \
    --model-path "$MODEL_PATH" --disaggregation-mode prefill \
    --extra-engine-args "$CONFIG_DIR/qwen3.5-gb300-agentic-prefill.yaml" \
    --publish-metrics
```

In the decode shell:

```bash
export CUDA_VISIBLE_DEVICES=1,2
export DYN_SYSTEM_PORT=8082
export TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1
mpirun --allow-run-as-root -np 2 --bind-to none \
    trtllm-llmapi-launch python3 -m dynamo.trtllm \
    --model-path "$MODEL_PATH" --disaggregation-mode decode \
    --extra-engine-args "$CONFIG_DIR/qwen3.5-gb300-agentic-decode.yaml" \
    --publish-metrics
```

If you also enable MTP (`speculative_config`) on the prefill worker, add
`export TRTLLM_DISAGG_ROLE=context` to the prefill shell. Ranks started by
`trtllm-llmapi-launch` do not receive a server role, so without it the prefill
worker reserves the GDN replay-state KV region that only decode uses. With
`pool_ratio` set, startup then fails with `initial_pool_ratio length must match
number of layer groups`. The profile above runs MTP on decode only and needs no
role setting.

The container runs as root; omit `--allow-run-as-root` when using a non-root user.
The rank counts must
match `tensor_parallel_size * pipeline_parallel_size` for each worker; launch a
separate MPI group for each replica. Use disjoint GPU masks and distinct system
ports for colocated workers. The source launcher binds worker host-memory
allocations to NUMA nodes `0,1`; on systems with that same NUMA layout, prefix
`trtllm-llmapi-launch` with `numactl -m 0,1`. Check the local CPU/NUMA topology
before applying that binding on another system.

The [prefill YAML](../../../examples/configs/curated/qwen3.5-gb300-agentic-prefill.yaml)
sets FP8 KV cache, conversation-scoped block reuse with three retained turns,
GDN state snapshots, chunked prefill, full-graph compilation, and piecewise CUDA
graphs. The [decode YAML](../../../examples/configs/curated/qwen3.5-gb300-agentic-decode.yaml)
sets MTP6, padded CUDA graphs, a no-eviction scheduler, and `stream_interval: 20`.
Both use KV cache manager V2, native host offload, NIXL with the Python
transceiver, and a 2048 MiB C++ transfer-agent bounce buffer for this
colocated profile. Its gate is relaxed (`agent_bounce_params`): each KV
write also carries the GDN state, whose large descriptors would otherwise
keep the write on standard NIXL.

Both workers use `--publish-metrics` with `enable_iter_perf_stats: true`,
`iter_perf_stats_interval: 40`, and
`kv_cache_config.iteration_stats_interval: 40`. Keep `return_perf_metrics: false`
and `kv_cache_config.event_buffer_max_size: 0`; metrics publication does not
require KV event publication. This profile uses session affinity with KV events
off. Send a stable `x-dynamo-session-id` for every turn of a conversation so the
frontend and attention-DP worker can preserve affinity. The router's idle TTL is
14400 seconds; reuse requires the retained cache entries to remain available.

### Select Another Topology

Start with copies of the two YAML files above and apply the following role
settings. All workers keep `pipeline_parallel_size: 1`, the common environment,
and 128 GiB of host cache per rank. `TP=EP` in every row. Attention DP is written
explicitly because equal TP/EP values alone do not determine attention behavior.

| Prefill profile | GPUs / TP / EP per worker | `enable_attention_dp` | `max_num_tokens` | Compile capture sizes |
|---|---:|---|---:|---|
| P1 | 1 | `true` | 8192 | Powers of two from 1 through 8192 |
| P4-TEP | 4 | `false` | 32768 | Powers of two from 1 through 32768 |
| P4-DEP | 4 | `true` | 8192 | Powers of two from 1 through 8192 |

Prefill retains `max_batch_size: 64`, `max_seq_len: 262144`,
`kv_cache_config.free_gpu_memory_fraction: 0.8`, and `moe_config.backend: CUTEDSL`.
Set `torch_compile_config.capture_num_tokens` to the complete list indicated
above. Keep `CUDA_SCALE_LAUNCH_QUEUES=4x` and
`TRTLLM_SKIP_KV_CACHE_ESTIMATION=1` on every prefill worker.

| Decode profile | GPUs / TP / EP per worker | Attention DP / LM-head TP in ADP | `max_batch_size` | `max_num_tokens` | CUDA graph `batch_sizes` | MoE backend |
|---|---:|---|---:|---:|---|---|
| D2-B8 | 2 | `false` / `false` | 8 | 56 | `[1, 2, 4, 8]` | `TRTLLM` |
| D2-B4 | 2 | `false` / `false` | 4 | 28 | `[1, 2, 4]` | `TRTLLM` |
| D8-B4 | 8 | `false` / `false` | 4 | 28 | `[1, 2, 4]` | `CUTEDSL` |
| D8-B1 | 8 | `false` / `false` | 1 | 7 | `[1, 2, 4]` | `CUTEDSL` |
| D16-B8 | 16 | `true` / `true` | 8 | 56 | `[1, 2, 4, 8]` | `CUTEDSL` |
| D4-B32 | 4 | `true` / `true` | 32 | 224 | `[1, 2, 4, 8, 16, 24, 32]` | `CUTEDSL` |

The two booleans set `enable_attention_dp` and `enable_lm_head_tp_in_adp`.
Decode retains `max_seq_len: 262148`, `kv_cache_config.avg_seq_len: 102150`,
`kv_cache_config.free_gpu_memory_fraction: 0.85`,
`moe_config.use_low_precision_moe_combine: true`, and MTP6.
The `avg_seq_len` is a cache-sizing input for the source traffic distribution;
revisit it when your conversation-length distribution changes.

| Topology | Prefill replicas × profile | Decode replicas × profile | Physical four-GPU nodes |
|---|---|---|---:|
| 1P1D, decode batch 8 | 1 × P1 | 1 × D2-B8, colocated | 1 |
| 1P1D, decode batch 4 | 1 × P1 | 1 × D2-B4, colocated | 1 |
| 1P2D | 1 × P4-TEP | 2 × D8-B4 | 5 |
| 1P4D, decode batch 1 | 1 × P4-TEP | 4 × D8-B1 | 9 |
| 1P4D, decode batch 4 | 1 × P4-TEP | 4 × D8-B4 | 9 |
| 2P3D | 2 × P4-TEP | 3 × D8-B4 | 8 |
| 3P1D | 3 × P4-DEP | 1 × D16-B8 | 7 |
| 3P2D | 3 × P4-DEP | 2 × D4-B32 | 5 |

For the two colocated 1P1D profiles, retain the bounce settings
(`cache_transceiver_config.kv_cache_bounce_size_mb: 2048` and
`agent_bounce_params`) on both roles and
`TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1` on both workers. For the other
profiles, remove those bounce settings and unset the weight-loading variable
to use the pinned runtime's defaults. Launch each TP8 worker across two
four-GPU nodes and each TP16 worker across four nodes, with one MPI rank per GPU.
Replicas share discovery endpoints, namespace, served model name, and the
frontend; do not combine separate replicas into a single MPI group. Pass the
same mounts and environment to every rank and use a cluster launcher to allocate
the indicated nodes before starting those MPI groups.

### Verify Responses and Metrics

Wait for worker initialization and model registration. Check readiness and the
registered model, then issue a chat request through Dynamo:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models
curl --fail http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'x-dynamo-session-id: qwen35-example-session' \
    -d '{"model":"nvidia/Qwen3.5-397B-A17B-NVFP4-V2","messages":[{"role":"user","content":"Write a Python function that adds two integers."}],"max_tokens":256}'
curl --fail http://localhost:8081/metrics
curl --fail http://localhost:8082/metrics
```

After traffic, verify that both worker endpoints expose
`trtllm_kv_cache_utilization` metrics. Send the full conversation history and the
same session header on subsequent turns. Measure latency, throughput, host-memory
use, and natural MTP acceptance on your own traffic before selecting a topology.
InferenceX runner setup, benchmark-client placement, trace selection, concurrency
sweeps, Enroot configuration, and synthetic acceptance are not required to run
these production serving commands.

## Qwen3.5 GB300 NVL72 AgentX Disaggregated Serving with Remote KV Cache

Use these profiles for long, multi-turn agentic **text** conversations with
`nvidia/Qwen3.5-397B-A17B-NVFP4-V2` on GB300 NVL72 with TensorRT LLM 1.4 and
Dynamo 1.8. Every profile runs TEP2 prefill workers with MTP1 and decode workers
with MTP6. Most profiles add the
[remote KV cache level](../features/kvcache.md#remote-cache-level-in-donated-host-memory-prototype):
decode ranks donate host memory they do not use, and prefill ranks keep more
reusable conversation prefixes in it, below their own host cache.

The serving settings come from [InferenceX #5147](https://github.com/SemiAnalysisAI/InferenceX/pull/5147)
(config `qwen3.5-fp4-gb300-dynamo-trt-agentic-disagg`), tested at commit
`3735ed9e27233597c1bb62508a5e3ceb2717bd99`. Its performance jobs use synthetic
speculative acceptance; these production commands leave acceptance to the model.
Those throughput results do not establish production throughput with natural
acceptance, or validate these Docker/MPI launch commands. To reproduce the
published points exactly, use the InferenceX harness at that commit.

### AgentX Profiles

Three profiles cover the operating range. Each has a prefill and a decode
engine-argument file in `examples/configs/curated`:

| Profile | Files | Prefill | Decode | KV cache below GPU | Admission and routing | Four-GPU nodes |
|---|---|---|---|---|---|---:|
| Remote-KV TEP8 | `qwen3.5-gb300-agentx-tep8-{prefill,decode}.yaml` | 1 × TEP2 | 1 × TEP8 (TP8, MoE TP2/EP4), batch 16 | Prefill: 168 GiB host per rank plus a 200 GiB remote level; decode donates 160 GiB per rank (fabric transport) | Priority by predicted reply length | 3 |
| High-throughput DEP8 | `qwen3.5-gb300-agentx-dep8-{prefill,decode}.yaml` | 4 × TEP2 | 1 × DEP8 (attention DP, EP8), batch 48 | Prefill: 168 GiB host per rank plus a 160 GiB remote level; decode donates 176 GiB per rank (NIXL transport) | First come, first served; least-loaded decode routing | 4 |
| Low-latency TEP16 | `qwen3.5-gb300-agentx-tep16-{prefill,decode}.yaml` | 1 × TEP2 | 1 × TEP16 (TP16, MoE TP2/EP8), batch 1 | 128 GiB host per rank on both roles; no remote level | Priority by predicted reply length | 5 |

All prefill workers use `tensor_parallel_size: 2` and
`moe_expert_parallel_size: 2` without attention DP, a 16384-token
`max_num_tokens`, breakable prefill CUDA graphs captured from 1 through 16384
tokens (`prefill_cuda_graph_backend: breakable`), `allreduce_strategy: TWOSHOT`,
the `CUTEDSL` MoE backend, NVFP4 GEMM backends restricted to `cutlass`,
`cublaslt`, `cutedsl`, and `cuda_core`, the `MAX_UTILIZATION` capacity scheduler
with `shortest_uncached_first` context scheduling, FP8 KV cache with 64-token
blocks and `pool_ratio: [0.2, 0.8]`, KV cache manager V2, and block reuse per
conversation with three retained turns and branch snapshots. They disable
generation CUDA graphs (`cuda_graph_config: null`) because they only run
context requests.

Decode workers keep `GUARANTEED_NO_EVICT` scheduling, padded CUDA graphs for
every batch size up to `max_batch_size` (a coarser list on DEP8),
`stream_interval: 20`, `replicate_embedding: true`, `max_seq_len: 262148`, and
`kv_cache_config.avg_seq_len: 102150`, a cache-sizing input for the source
traffic; revisit it when your conversation lengths differ. The TEP decode
workers use the `TRTLLM` MoE backend with low-precision combine and keep the
explicit transfer budget `cache_transceiver_config.max_tokens_in_buffer:
4194304`; without it, the generation side skips its transfer window. The DEP8
worker uses the `CUTEDSL` MoE backend, LM-head TP in attention DP, and
`free_gpu_memory_fraction: 0.85`. With priority admission
(`waiting_queue_policy: priority`), a request that has waited
`waiting_queue_max_wait_s` goes ahead of newer ones; keep that value at or below
the prefill `kv_transfer_timeout_ms` (600 s). Those profiles also release
finished requests' KV cache one step later, overlapping the release with GPU work
(`defer_generation_only_release: true`).

### AgentX Runtime and Host-Memory Budget

Use the exact versions together:

| Component | Version |
|---|---|
| TensorRT LLM container | `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc4.dev202611170009` |
| `ai-dynamo` and `ai-dynamo-runtime` wheels | `1.8.0.dev20261119` |
| Checkpoint | `nvidia/Qwen3.5-397B-A17B-NVFP4-V2` |

The configuration files must come from your TensorRT LLM checkout; the container
does not bundle them. Requalify the checkpoint, configuration, and transport
together when upgrading.

Host memory bounds these profiles. Each rank reserves its host cache, and each
donating rank holds its donation for the life of its process:

| Node role | Reservation per four-GPU node |
|---|---:|
| Two TEP2 prefill workers, 168 GiB host cache per rank | 672 GiB |
| TEP8 decode ranks donating 160 GiB each | 640 GiB |
| DEP8 decode ranks donating 176 GiB each | 704 GiB |
| Host-cache profile ranks, 128 GiB each | 512 GiB |

The source system allocates 898 GiB of host memory to each four-GPU node, so the
remainder covers weights, loading, transfer staging, the frontend, and the
operating system. A prefill rank's remote level lives in the donors' memory, not
on its own node, but the donors' published memory must cover every claim. Check
the budget of your own nodes before raising a cache size or a donation.

Use a GB300 NVL72 system with working GPU peer access and NIXL/UCX transport. The
fabric transport needs prefill and decode nodes in the same multi-node NVLink
domain. The DEP8 profile uses the NIXL transport (RDMA) for the remote level
instead: on the source cluster, host-memory fabric handles exported from some
nodes could not be imported, and its 4P1D layout has too little spare donation
to lose one decode node's donors. Every rank that shares a rendezvous directory
must use the same transport. Have reachable etcd and NATS services before
starting Dynamo; see the
[Dynamo CLI installation guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/cli/installation/install-dynamo.mdx).

The remote level is a prototype. Prefill workers use tensor parallelism without
attention DP, so if a donor process exits, the claiming prefill workers stop
with `RemoteMemError`. Start and stop the prefill and decode workers that share
a rendezvous directory together, without independent restarts or scale-down of
the donating workers. See the
[failure behavior](../features/kvcache.md#remote-cache-level-in-donated-host-memory-prototype)
of the remote level before deploying it.

### Prepare the AgentX Container

Download the exact V2 checkpoint into a local directory on every node. Run the
following from the TensorRT LLM checkout on each node, replacing `MODEL_DIR` and
`SHARED_DIR`. `SHARED_DIR` must be a directory on a file system that every node
sees and that only you can write:

```bash
export MODEL_DIR=/absolute/path/to/Qwen3.5-397B-A17B-NVFP4-V2
export SHARED_DIR=/absolute/shared/path/qwen35-agentx
docker run --rm -it --name qwen35-agentx \
    --gpus all --network host --ipc host --ulimit memlock=-1 \
    --mount type=bind,src="$MODEL_DIR",dst=/models/qwen35,readonly \
    --mount type=bind,src="$PWD",dst=/opt/trtllm-recipes,readonly \
    --mount type=bind,src="$SHARED_DIR",dst=/shared \
    nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc4.dev202611170009 \
    /bin/bash
```

Inside each container, install the matching public wheels:

```bash
python3 -m pip install --pre \
    --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com \
    ai-dynamo==1.8.0.dev20261119 ai-dynamo-runtime==1.8.0.dev20261119
python3 -c 'import importlib.metadata as m; print({p: m.version(p) for p in ("tensorrt-llm", "ai-dynamo", "ai-dynamo-runtime")})'
```

On one node, write the Dynamo router policy that the frontend and the decode
workers of the priority-admission profiles share:

```bash
cat > /shared/router-policy.yaml <<'EOF'
request_classifier:
  type: output_length_prediction
EOF
```

With it, the Dynamo KV router predicts each request's reply length and forwards
it as the request's TensorRT LLM priority, which decode workers with
`waiting_queue_policy: priority` admit by.

Open a shell in the container for each of the frontend, prefill, and decode
commands below with `docker exec -it qwen35-agentx /bin/bash`. Run this setup in
**each shell**, and replace the service endpoints when they run on another host:

```bash
export MODEL_PATH=/models/qwen35
export CONFIG_DIR=/opt/trtllm-recipes/examples/configs/curated
export ETCD_ENDPOINTS=http://127.0.0.1:2379
export NATS_SERVER=nats://127.0.0.1:4222
export DYN_NAMESPACE=qwen35-agentx
export DYN_DISCOVERY_BACKEND=etcd
export DYN_REQUEST_PLANE=tcp
export KV_RENDEZVOUS_DIR=/shared/kv-rendezvous/deployment-001
source "$CONFIG_DIR/qwen3.5-gb300-agentx.env"
```

`remote_cache_config.directory` in the configuration files is
`${KV_RENDEZVOUS_DIR}`; TensorRT LLM expands it and refuses to start when it is
unset. Use a new directory for each deployment, shared by exactly the workers that
start and stop together. Donor files hold memory registration keys or fabric
handles, so TensorRT LLM creates the directory with mode 0700 and rejects one that
another user owns or can write.

The [worker environment](../../../examples/configs/curated/qwen3.5-gb300-agentx.env)
sets the source profile's allocator, CUDA launch, garbage-collection,
weight-staging, transfer, MPI collective, and UCX polling settings. It includes
an intentionally empty `MIMALLOC_PURGE_DELAY`. Keep the draft head and its
precision as shipped with the V2 checkpoint. Never set
`TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` in production or accuracy runs.

On the source nodes, GPU dma-buf registration failed because each GPU and NIC
sat in different IOMMU groups. If NIXL fails to register GPU memory on your
system for that reason, also export `UCX_IB_DIRECT_NIC=n` and
`UCX_CUDA_COPY_DMABUF=no` in the worker shells, so that UCX registers GPU memory
through `nvidia_peermem` instead. Leave them unset on systems where dma-buf
registration works.

### Start the AgentX Frontend and Workers

The commands below start the remote-KV TEP8 profile. In the frontend shell:

```bash
export ETCD_LEASE_TTL=120
export DYN_TOKENIZER_CACHE=1
export DYN_TOKENIZER_CACHE_BYTES=8000000000
export DYN_TOKENIZER=fastokens
export DYN_TOKEN_IDS_AS_BYTES=1
python3 -m dynamo.frontend --http-port 8000 \
    --router-mode kv --router-session-affinity-ttl-secs 14400 \
    --router-policy-config /shared/router-policy.yaml
```

The frontend builds its router plugins from its own policy at startup, so start
it with the policy before the decode workers that advertise the classifier.

In the prefill shell, on a prefill node:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export DYN_SYSTEM_PORT=8081
export TRTLLM_DISAGG_ROLE=context
export CUDA_SCALE_LAUNCH_QUEUES=4x
export TRTLLM_SKIP_KV_CACHE_ESTIMATION=1
export NCCL_MIN_NCHANNELS=64
export NCCL_MAX_NCHANNELS=64
export UCX_RC_VERBS_TIMEOUT=4s
export UCX_KEEPALIVE_INTERVAL=60s
export DYN_ENGINE_CONV_AFFINITY=1
export DYN_ROUTER_MODE=kv
export DYN_ROUTER_SESSION_AFFINITY_TTL_SECS=14400
mpirun --allow-run-as-root -np 2 --bind-to none \
    trtllm-llmapi-launch python3 -m dynamo.trtllm \
    --model "$MODEL_PATH" --disaggregation-mode prefill \
    --extra-engine-args "$CONFIG_DIR/qwen3.5-gb300-agentx-tep8-prefill.yaml" \
    --publish-metrics
```

In the decode shell, for one TEP8 worker across two decode nodes:

```bash
export DYN_SYSTEM_PORT=8082
export UCX_RC_VERBS_TIMEOUT=4s
export UCX_KEEPALIVE_INTERVAL=60s
export DYN_ROUTER_MODE=kv
export DYN_ROUTER_POLICY_CONFIG=/shared/router-policy.yaml
mpirun --allow-run-as-root -np 8 --bind-to none \
    trtllm-llmapi-launch python3 -m dynamo.trtllm \
    --model "$MODEL_PATH" --disaggregation-mode decode \
    --extra-engine-args "$CONFIG_DIR/qwen3.5-gb300-agentx-tep8-decode.yaml" \
    --publish-metrics
```

`TRTLLM_DISAGG_ROLE=context` is required on every prefill worker of these
profiles. Ranks started by `trtllm-llmapi-launch` do not receive a server role;
without it, a prefill worker with MTP also reserves the GDN replay-state region
that only decode uses, and with `pool_ratio` set, startup fails with
`initial_pool_ratio length must match number of layer groups`.
`CUDA_SCALE_LAUNCH_QUEUES`, `TRTLLM_SKIP_KV_CACHE_ESTIMATION`, and the NCCL
channel counts are prefill-only settings of the source profile.
`DYN_ENGINE_CONV_AFFINITY=1` forces the TensorRT LLM worker's
conversation-affinity routing on regardless of engine-config detection. The UCX keep-alive and RC timeout settings apply to the
remote-KV profiles; the source does not set them on the host-cache profile.

The decode workers' `DYN_ROUTER_MODE` advertises a router configuration that
overrides the frontend's routing for the decode worker set only; the frontend
itself always runs with `--router-mode kv`. `DYN_ROUTER_POLICY_CONFIG` must match
the frontend's `--router-policy-config`: a decode worker set that advertises the
classifier fails to register unless the frontend has built it too, so set both or
neither. Start the donating decode workers
and the claiming prefill workers together. A claimant waits up to 900 seconds
(`level.donor_wait_timeout_s`) for the donors in `level.expected`, and fails at
once when the reported donors cannot cover the claims.

The container runs as root; omit `--allow-run-as-root` when using a non-root user.
The rank count of each worker must match
`tensor_parallel_size * pipeline_parallel_size`; launch a separate MPI group for
each replica, and do not combine replicas into a single MPI group. Use disjoint
GPU masks and distinct system ports for colocated workers. Run each TP8 worker
across two four-GPU nodes and each TP16 worker across four, with one rank per
GPU, and start every rank with the same mounts and environment, for example with
a cluster launcher that propagates the environment or Open MPI's `-x NAME` for
each exported variable. Replicas share the discovery endpoints, namespace, served
model name, and frontend.

The source launcher runs each worker rank on the CPUs of its GPU's NUMA node and
prefers that node's memory (`numactl --cpunodebind=<node> --preferred=<node>`),
and keeps `TLLM_NUMA_AWARE_WORKER_AFFINITY=1`. To do the same, prefix
`trtllm-llmapi-launch` with a per-rank wrapper that resolves the node from the
rank's GPU, for example:

```bash
cat > /shared/numa-preferred.sh <<'EOF'
#!/bin/bash
set -euo pipefail
IFS=, read -ra gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
gpu=${gpus[${OMPI_COMM_WORLD_LOCAL_RANK:-0}]}
bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$gpu")
node=$(cat "/sys/bus/pci/devices/$(printf '%04x:%s' "0x${bus%%:*}" "${bus#*:}" | tr 'A-F' 'a-f')/numa_node")
exec numactl --cpunodebind="$node" --preferred="$node" "$@"
EOF
chmod +x /shared/numa-preferred.sh
```

Check the NUMA topology of your system before applying that binding.

Both roles use `--publish-metrics` with `enable_iter_perf_stats: true`,
`iter_perf_stats_interval: 40`, and
`kv_cache_config.iteration_stats_interval: 40`. Keep `return_perf_metrics: false`
and `kv_cache_config.event_buffer_max_size: 0`; metrics publication does not
require KV event publication. These profiles use session affinity with KV events
off. Send a stable `x-dynamo-session-id` for every turn of a conversation. The
router's idle TTL is 14400 seconds; reuse requires the retained cache entries to
remain available.

To run the other profiles, replace the configuration files and apply these
role differences:

* **High-throughput DEP8:** use `qwen3.5-gb300-agentx-dep8-{prefill,decode}.yaml`,
  start four TEP2 prefill workers, two per prefill node, and one eight-rank decode
  worker across two nodes. Start the frontend with `--router-mode kv` and without
  `--router-policy-config`, and set `DYN_ROUTER_MODE=least-loaded` without
  `DYN_ROUTER_POLICY_CONFIG` on the decode worker, which overrides the frontend's
  KV routing for the decode worker set. Both roles use the 2048 MiB transfer-agent bounce buffer
  (`kv_cache_bounce_size_mb: 2048`) with
  `agent_bounce_params.max_average_descriptor_size: 64KB`.
* **Low-latency TEP16:** use `qwen3.5-gb300-agentx-tep16-{prefill,decode}.yaml`,
  start one TEP2 prefill worker and one sixteen-rank decode worker across four
  nodes, and leave `UCX_RC_VERBS_TIMEOUT` and `UCX_KEEPALIVE_INTERVAL` unset on
  both roles. `KV_RENDEZVOUS_DIR` is not used.

### Map the InferenceX Points

InferenceX #5147 measured thirteen points. Each one uses the files of the
profile listed below with the stated changes; unlisted settings match the
profile's files. "Base graphs" means CUDA graph batch sizes 1 through 16, then
even sizes through 32, then 40, 48, 56, and 64. On remote points, set the
prefill `kv_cache_config.remote_cache_config.level.expected` to the job's
donating rank count and the total claim of all prefill ranks shown (the source
rounded each 200 GiB claim up to 203 GiB and each 160 GiB claim to 162 GiB).

| Point (prefill × TEP2, decode workers) | InferenceX concurrency | Profile files | Decode `max_batch_size` and CUDA graphs | Decode admission | `level.expected` (`num_donors`, `claim_size`) | Nodes |
|---|---:|---|---|---|---|---:|
| 1P1D, 1 × TEP8 | 240 | TEP8, unchanged | 16, sizes 1-16 | Priority, 120 s cap | 8, 435939180544 | 3 |
| 1P2D, 2 × TEP8 | 48 | TEP8 | 4, sizes 1-4 | Priority, 60 s cap | 16, 435939180544 | 5 |
| 1P2D, 2 × TEP8 | 200 | TEP8 | 5, sizes 1-5 | Priority, 120 s cap | 16, 435939180544 | 5 |
| 2P1D, 1 × TEP8 | 192 | TEP8 | 64, base graphs | FCFS, least-loaded, no deferred release | 8, 871878361088 | 3 |
| 2P1D, 1 × TEP8 | 352 | TEP8 | 64, base graphs | Priority, 60 s cap, no deferred release | 8, 871878361088 | 3 |
| 2P1D, 1 × TEP8 | 416 | TEP8 | 64, base graphs | Priority, 60 s cap, no deferred release | 8, 871878361088 | 3 |
| 2P3D, 3 × TEP8 | 384 | TEP8 | 8, sizes 1-8 | Priority, 60 s cap | 24, 871878361088 | 7 |
| 3P2D, 2 × TEP8 | 456 | TEP8 | 64, base graphs | FCFS, least-loaded, no deferred release | 16, 1307817541632 | 6 |
| 3P2D, 2 × TEP8 | 512 | TEP8 | 32, base graphs through 32 | FCFS, least-loaded, no deferred release | 16, 1307817541632 | 6 |
| 4P1D, 1 × DEP8 | 840 | DEP8, unchanged | 48, as in the file | FCFS, least-loaded | 8, 1391569403904 | 4 |
| 1P1D, 1 × TEP16 | 32 | TEP16, unchanged | 1, size 1 | Priority, 120 s cap | Not used | 5 |
| 1P2D, 2 × TEP8 | 96 | TEP8 decode, TEP16 prefill | 2, sizes 1-2 | Priority, 120 s cap | Not used | 5 |
| 1P4D, 4 × TEP8 | 96 | TEP8 decode, TEP16 prefill | 1, size 1 | Priority, 60 s cap | Not used | 9 |

"FCFS" removes `waiting_queue_policy` and `waiting_queue_max_wait_s`; "no
deferred release" removes `defer_generation_only_release`. On FCFS points, start
the frontend with `--router-mode kv` and without `--router-policy-config`, and set
`DYN_ROUTER_MODE=least-loaded` without `DYN_ROUTER_POLICY_CONFIG` on the decode
workers, which overrides the frontend's KV routing for the decode worker set. On the last two points, the decode workers replace
`remote_cache_config` with `host_cache_size: 137438953472` (128 GiB) and leave
the UCX keep-alive and RC timeout variables unset, as in the TEP16 profile.
Two TEP2 prefill workers share a node; the node counts exclude the frontend,
which shares a decode node. Per-point engine-argument and environment files
generated from the InferenceX recipe are proposed in
[#20788](https://github.com/NVIDIA/TensorRT-LLM/pull/20788) (pending).

### Verify the AgentX Deployment

Wait for worker initialization and model registration. Check readiness and the
registered model, then issue a chat request through Dynamo:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/v1/models
curl --fail http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'x-dynamo-session-id: qwen35-agentx-example-session' \
    -d '{"model":"nvidia/Qwen3.5-397B-A17B-NVFP4-V2","messages":[{"role":"user","content":"Write a Python function that adds two integers."}],"max_tokens":256}'
curl --fail http://localhost:8081/metrics
curl --fail http://<decode-leader-node>:8082/metrics
```

After traffic, verify that both worker endpoints expose
`trtllm_kv_cache_utilization` metrics. With `TLLM_LOG_LEVEL=INFO`, each rank logs
the KV cache tiers it built: decode ranks of remote profiles log their donation,
for example `KV cache manager v2 donates 160.00GiB of host memory to remote cache
levels`, and prefill ranks log `KV cache manager v2 remote cache quota set to
200.00GiB`. Send the full conversation history and the same session header on
subsequent turns. Measure latency, throughput, host-memory use, and natural MTP
acceptance on your own traffic before selecting a profile. InferenceX runner
setup, benchmark-client placement, trace selection, concurrency sweeps, and
synthetic acceptance are not required to run these production serving commands.

## Testing API Endpoint

### Basic Test

Start a new terminal on the host to test the TensorRT LLM server you just launched.

You can query the health/readiness of the server using:

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When the `Status: 200` code is returned, the server is ready for queries. Note that the very first query may take longer due to initialization and compilation.

After the TensorRT LLM server is set up and shows Application startup complete, you can send requests to the server.

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json"  -d '{
    "model": "<model_path_or_hf_id>",
    "messages": [
        {
            "role": "user",
            "content": "Where is New York?"
        }
    ],
    "max_tokens": 1024,
    "top_p": 1.0
}' -w "\n"
```

## Benchmarking Performance

To benchmark the performance of your TensorRT LLM server you can leverage the built-in `benchmark_serving.py` script. To do this, first create a wrapper `bench.sh` script.

```shell
cat <<'EOF' > bench.sh
#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="<model_path_or_hf_id>"

concurrency_list="1 2 4 8 16 32 64 128 256"
multi_round=5
isl=8192
osl=1024
result_dir=/tmp/qwen3_output

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model ${MODEL_NAME} \
        --backend openai \
        --dataset-name "random" \
        --random-input-len ${isl} \
        --random-output-len ${osl} \
        --random-prefix-len 0 \
        --random-ids \
        --num-prompts ${num_prompts} \
        --max-concurrency ${concurrency} \
        --ignore-eos \
        --tokenize-on-client \
        --percentile-metrics "ttft,tpot,itl,e2el"
done
EOF
chmod +x bench.sh
```

To achieve max throughput, with attention DP on, one needs to sweep up to `concurrency = max_batch_size * num_gpus`.

If you want to save the results to a file add the following options.

```shell
--save-result \
--result-dir "${result_dir}" \
--result-filename "concurrency_${concurrency}.json"
```

For more benchmarking options see [benchmark_serving.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py)

Run `bench.sh` to begin a serving benchmark. This will take a long time if you run all the concurrencies mentioned in the above `bench.sh` script.

```shell
./bench.sh
```

## Key Configuration Options

These options control TensorRT LLM behavior and are set in the YAML file passed to `trtllm-serve` with the `--config` argument.

### `tensor_parallel_size`

Sets the tensor-parallel size. This should typically match the number of GPUs used by one model instance.

### `moe_expert_parallel_size`

Sets the expert-parallel size for MoE layers. It can differ from the attention tensor-parallel behavior when attention data parallelism is enabled.

### `enable_attention_dp`

Enables data parallelism for attention and linear-attention layers while keeping the MoE layers expert-parallel. This is primarily a throughput-oriented configuration.

### `kv_cache_config.free_gpu_memory_fraction`

Specifies the fraction of free GPU memory reserved for the KV cache after the model is loaded. Reduce this value if initialization reports an out-of-memory error.

### `kv_cache_config.use_kv_cache_manager_v2`

Selects KV cache manager V2 when set to `true`. Preserve the value in the curated profile unless the alternative cache-management path has been qualified with the selected topology and MTP mode.

### `kv_cache_config.avg_seq_len`

Helps KV cache manager V2 divide memory between attention KV blocks and recurrent GDN state. Set it to the expected average total sequence length for throughput deployments.

### `kv_cache_config.mamba_ssm_cache_dtype`

Selects the GDN recurrent-state dtype independently of the attention KV-cache dtype. Supported values are `auto`, `float16`, `bfloat16`, and `float32`.

### `max_batch_size`

Sets the maximum number of requests that can be grouped into one scheduled batch. The achievable batch size also depends on the total input and output sequence lengths.

### `max_num_tokens`

Sets the maximum total number of tokens across all requests in one scheduled batch.

### `max_seq_len`

Sets the maximum sequence length for one request, including input and generated tokens.

### `trust_remote_code`

Allows Hugging Face to load custom model and tokenizer code from the model repository. Enable it only for trusted model sources.

### `cuda_graph_config`

Controls CUDA graph capture and padding:

* `enable_padding`: Pads input batches to a captured CUDA graph batch size.
* `max_batch_size`: Sets the largest batch size for which CUDA graphs are captured.

### `moe_config`

Controls MoE execution:

* `backend`: Selects the MoE backend.
* `max_num_tokens`: Limits the number of tokens processed by one fused MoE invocation before chunking.
* `load_balancer`: Accepts a static EPLB configuration file or an inline Online EPLB configuration.

### `speculative_config.max_draft_len`

Sets the MTP draft length. Tune it for the deployment objective; the MTP3 profiles set it to 3.

See the [`TorchLlmArgs` API reference](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the complete configuration schema, [KV cache documentation](../features/kvcache.md) for hybrid-state cache settings, and [speculative decoding documentation](../features/speculative-decoding.md) for MTP details.

## Troubleshooting Tips

* If you encounter CUDA out-of-memory errors, try reducing `max_batch_size`, `max_num_tokens`, or `kv_cache_config.free_gpu_memory_fraction`. If the error occurs during CUDA graph capture, also reduce `cuda_graph_config.max_batch_size`.
* Ensure your model checkpoints are compatible with the expected format. If a Qwen3.8 MoE checkpoint is routed to an unsupported architecture, verify that its `config.json` advertises `Qwen3_5MoeForCausalLM` and `qwen3_5_moe_text`.
* If MTP output is empty or incorrect, confirm that the checkpoint contains its MTP weights and that the selected `max_draft_len` is configured consistently on every rank.
* For performance issues, check GPU utilization with `nvidia-smi` while the server is running and complete initialization and warmup before measuring performance.
* If static EPLB regresses throughput, verify the placement-file hash and `num_slots`, and recollect statistics using representative traffic. Do not reuse a map from another checkpoint or MTP mode.
* If online EPLB is enabled, wait for the configured observation and migration period before measuring steady state, and retain logs that show completed layer updates.
* If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed.
* For connection issues, make sure the server port (`8000` in this guide) is not being used by another application.
* Reasoning is controlled with `--reasoning_parser qwen3_5`. To toggle thinking per request, pass `enable_thinking` through `chat_template_kwargs` in the request body, for example `{"chat_template_kwargs": {"enable_thinking": true}}` (set it to `false` to disable reasoning).
