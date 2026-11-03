<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment Guide for Kimi K3.1 Pro on TensorRT LLM - Blackwell (GB300)

## Introduction

This guide describes how to serve Kimi K3.1 Pro with TensorRT LLM on NVIDIA GB300 NVL72 systems. It covers obtaining the checkpoint and its format, the supported deployments, building the software, the configuration files, launching the disaggregated servers (context (prefill) servers, a generation (decode) server and a router), running the model through the LLM API, the model-specific API behavior, example requests, accuracy verification and the known limitations.

Kimi K3.1 Pro builds on Kimi K3, a hybrid Mixture-of-Experts (MoE) model with 896 routed experts. Most decoder layers use Kimi Delta Attention (KDA), a linear-attention mixer that keeps a constant-size recurrent state per sequence; every fourth layer uses Multi-head Latent Attention (MLA). In Kimi K3.1 Pro the MLA layers add a DeepSeek Sparse Attention (DSA) indexer that selects the tokens each query attends to. The routed experts are stored in MXFP4. The checkpoint also contains a vision encoder. TensorRT LLM serves it as the architecture `KimiK31ForConditionalGeneration` (`model_type: kimi_k31`).

## Prerequisites

* GPU: NVIDIA GB300 (`sm_103`) in an NVL72 rack. Each disaggregated deployment runs entirely in one rack, so that the KV cache moves between servers over NVLink. A tray has 4 GPUs.
* Multi-node launcher: Slurm with the pyxis/enroot container plugin and PMIx (`srun --mpi=pmix`), or an equivalent MPI launcher, to start one rank per GPU, and a container runtime with NVIDIA GPU support on each tray.
* A shared filesystem visible to all trays for the repository, the checkpoint, the DSpark draft model, and the configuration and environment files.
* Host memory: the context servers offload KV cache blocks to host memory (`kv_cache_config.host_cache_size`: 160 GiB per rank on the TEP8 context servers, 150 GiB per rank on the DEP16 context servers). Lower it if the trays have less free host memory.

## Models

* **Kimi K3.1 Pro checkpoint**, [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro), in Hugging Face format. This is the model TensorRT LLM serves.
* **DSpark draft model**: the Kimi K3 DSpark draft, [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark), a separate checkpoint used for speculative decoding. Support for a Kimi K3.1 Pro DSpark draft model will be added later.

| Checkpoint | Weights | Features | Deployments |
| :-- | :-- | :-- | :-- |
| Kimi K3.1 Pro | Routed experts pre-quantized to MXFP4; BF16 attention, shared expert and vision encoder | Text, images (DEP16 deployments), DSpark speculative decoding, contexts up to 1,048,576 tokens | All [supported configurations](#supported-configurations) |
| Kimi K3 DSpark draft | Draft layers of the Kimi K3 DSpark model | DSpark speculative decoding | All disaggregated deployments |

### Download the Checkpoint

Download the Kimi K3.1 Pro checkpoint, [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro), and the DSpark draft model, [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark), once to the shared filesystem. Pass both directories to TensorRT LLM as they are; no conversion or quantization step is needed. The examples below use `<KIMI_K31_MODEL_DIR>` and `<DSPARK_DRAFT_MODEL_DIR>`; replace them with your local paths.

```bash
huggingface-cli download moonshotai/Kimi-K3.1-Pro --local-dir <KIMI_K31_MODEL_DIR>
huggingface-cli download Inferact/Kimi-K3-DSpark --local-dir <DSPARK_DRAFT_MODEL_DIR>
```

### Checkpoint Format

The checkpoint is in the Hugging Face format: `config.json` (`model_type: kimi_k31`), the weights, and the tokenizer and chat template together with their Python code. TensorRT LLM reads the model configuration, including the MXFP4 quantization of the routed experts, from `config.json` and loads the architecture `KimiK31ForConditionalGeneration`.

All configurations set `trust_remote_code: true`, which the checkpoint's configuration and tokenizer code need. The DSpark draft model is passed separately, as `speculative_config.speculative_model`.

## Supported Configurations

Kimi K3.1 Pro supports:

* **Disaggregated serving** with TEP or DEP context and generation servers.
* **Aggregated serving** with a TEP or DEP server.

The server layouts are TEP8 (`tensor_parallel_size: 8`, `moe_expert_parallel_size: 8`), TEP16 (`tensor_parallel_size: 16`, `moe_expert_parallel_size: 4`, `moe_tensor_parallel_size: 4`), TEP32 (`tensor_parallel_size: 32`, `moe_expert_parallel_size: 8`, `moe_tensor_parallel_size: 4`) and DEP16 (`tensor_parallel_size: 16`, `moe_expert_parallel_size: 16`, `enable_attention_dp: true`). The TEP servers store the KV cache and the indexer keys in FP8; the DEP16 servers use `nvfp4_ds_mla` with FP4 indexer keys.

Numerics shared by all deployments: BF16 attention, shared expert and collectives; routed experts with MXFP4 weights and MXFP8 activations (W4A8) on the TRTLLM-Gen MoE backend; FP32 KDA recurrent state; FP8 storage for the DSpark draft's context KV. All servers set `max_seq_len: 1048576`.

Aggregated DEP16 runs one server on 4 trays with `kv_cache_config.free_gpu_memory_fraction: 0.5` and `moe_config.max_num_tokens: 32768`; its KV cache holds about 500k tokens per request. See [Quick Start Without a Server (LLM API)](#quick-start-without-a-server-llm-api) and [Known Limitations](#known-limitations).

## Deployment Steps

### Build from Source

Inside the TensorRT LLM development container (SBSA variant on GB300), from the repository root:

```bash
python3 scripts/build_wheel.py --cuda_architectures 103-real --skip_building_wheel --yes
.venv-3.12/bin/python -m pip install -e .
```

`build_wheel.py` creates the virtual environment `.venv-3.12` at the repository root (named after the container's Python version). The installation pulls in `fla-core` and `einops` from `requirements.txt`, which the KDA layers need. The launch commands below run from this in-place environment, so build with the repository at the same path the servers use. See [Building from Source](../installation/build-from-source.md) for other options.

### Configuration Files

Each server layout has one configuration file, and the router has its own. Save the files of a deployment in one directory, replace `<DSPARK_DRAFT_MODEL_DIR>` with the DSpark draft checkpoint, and replace each `<..._HOST>` in `router.yaml` with the rank-0 tray of that server. A deployment uses the files of its context and generation servers, with the changes listed under a file for some pairings.

#### TEP8 Context Server

`tep8_ctx.yaml`:

```yaml
tensor_parallel_size: 8
moe_expert_parallel_size: 8
enable_attention_dp: false
trust_remote_code: true
disable_overlap_scheduler: true
cuda_graph_config:
  max_batch_size: 1
disable_mm_encoder: true
prefill_cuda_graph_backend: breakable
prefill_capture_num_tokens: [256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096]
enable_chunked_prefill: true
max_batch_size: 8
max_num_tokens: 16384
max_seq_len: 1048576
print_iter_log: true
stream_interval: 10
num_serve_frontends: 8
kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: fp8
  tokens_per_block: 64
  enable_block_reuse: true
  block_reuse_config:
    policy: per_conversation
    max_num_turns: 5
  mamba_state_config:
    additional_snapshot_offsets_from_end: [0]
  avg_seq_len: 150000
  free_gpu_memory_fraction: 0.7
  host_cache_size: 171798691840
moe_config:
  max_num_tokens: 33024
speculative_config:
  decoding_type: DSpark
  max_draft_len: 5
  block_size: 5
  speculative_model: <DSPARK_DRAFT_MODEL_DIR>
  context_kv_dtype: fp8
  context_kv_only: true
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  kv_cache_bounce_size_mb: 0
  kv_transfer_timeout_ms: 600000
sparse_attention_config:
  algorithm: dsa
  q_split_threshold: 1024
  enable_heuristic_topk: true
  gvr_deterministic_topk: true
```

With a TEP32 generation server, set `speculative_config.max_draft_len: 7` and `speculative_config.block_size: 7`.

#### TEP16 Generation Server

`tep16_gen.yaml`:

```yaml
tensor_parallel_size: 16
moe_expert_parallel_size: 4
enable_attention_dp: false
moe_tensor_parallel_size: 4
trust_remote_code: true
max_batch_size: 32
max_num_tokens: 192
max_seq_len: 1048576
print_iter_log: true
stream_interval: 100
disable_overlap_scheduler: false
num_serve_frontends: 8
num_postprocess_workers: 4
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: fp8
  tokens_per_block: 64
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.8
speculative_config:
  decoding_type: DSpark
  max_draft_len: 5
  block_size: 5
  speculative_model: <DSPARK_DRAFT_MODEL_DIR>
  context_kv_dtype: fp8
  advanced_sampling_mode: full
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  kv_cache_bounce_size_mb: 0
  kv_transfer_timeout_ms: 600000
sparse_attention_config:
  algorithm: dsa
  enable_heuristic_topk: true
  gvr_deterministic_topk: true
```

With 2 x TEP8 context servers, set `max_batch_size: 48`, `max_num_tokens: 288` and `cuda_graph_config.batch_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]`.

#### TEP32 Generation Server

`tep32_gen.yaml`:

```yaml
tensor_parallel_size: 32
moe_expert_parallel_size: 8
enable_attention_dp: false
moe_tensor_parallel_size: 4
trust_remote_code: true
max_batch_size: 32
max_num_tokens: 192
max_seq_len: 1048576
print_iter_log: true
stream_interval: 100
disable_overlap_scheduler: false
num_serve_frontends: 8
num_postprocess_workers: 4
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: fp8
  tokens_per_block: 64
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.8
speculative_config:
  decoding_type: DSpark
  max_draft_len: 7
  block_size: 7
  speculative_model: <DSPARK_DRAFT_MODEL_DIR>
  context_kv_dtype: fp8
  advanced_sampling_mode: full
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  kv_cache_bounce_size_mb: 0
  kv_transfer_timeout_ms: 600000
sparse_attention_config:
  algorithm: dsa
  enable_heuristic_topk: true
  gvr_deterministic_topk: true
```

#### DEP16 Context Server

`dep16_ctx.yaml`:

```yaml
tensor_parallel_size: 16
moe_expert_parallel_size: 16
enable_attention_dp: true
trust_remote_code: true
disable_overlap_scheduler: true
cuda_graph_config: null
enable_chunked_prefill: true
max_batch_size: 16
max_num_tokens: 8192
max_seq_len: 1048576
print_iter_log: true
stream_interval: 10
num_serve_frontends: 8
kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: nvfp4_ds_mla
  tokens_per_block: 64
  enable_block_reuse: true
  block_reuse_config:
    policy: per_conversation
    max_num_turns: 3
  mamba_state_config:
    additional_snapshot_offsets_from_end: [0]
  avg_seq_len: 150000
  free_gpu_memory_fraction: 0.7
  host_cache_size: 161061273600
moe_config:
  max_num_tokens: 65536
attention_dp_config:
  kv_cache_routing_conversation_affinity: true
  kv_cache_routing_new_conv_placement: least_queued
speculative_config:
  decoding_type: DSpark
  max_draft_len: 6
  block_size: 6
  speculative_model: <DSPARK_DRAFT_MODEL_DIR>
  context_kv_dtype: fp8
  context_kv_only: true
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  kv_cache_bounce_size_mb: 0
  kv_transfer_timeout_ms: 600000
sparse_attention_config:
  indexer_k_dtype: fp4
  algorithm: dsa
  enable_heuristic_topk: true
  gvr_deterministic_topk: true
```

#### DEP16 Generation Server

`dep16_gen.yaml`:

```yaml
tensor_parallel_size: 16
moe_expert_parallel_size: 16
enable_attention_dp: true
trust_remote_code: true
max_batch_size: 32
max_num_tokens: 256
max_seq_len: 1048576
print_iter_log: true
stream_interval: 100
disable_overlap_scheduler: false
num_serve_frontends: 8
num_postprocess_workers: 4
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: nvfp4_ds_mla
  tokens_per_block: 64
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.9
speculative_config:
  decoding_type: DSpark
  max_draft_len: 6
  block_size: 6
  speculative_model: <DSPARK_DRAFT_MODEL_DIR>
  context_kv_dtype: fp8
  advanced_sampling_mode: no_topk
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  kv_cache_bounce_size_mb: 0
  kv_transfer_timeout_ms: 600000
sparse_attention_config:
  indexer_k_dtype: fp4
  algorithm: dsa
  enable_heuristic_topk: true
  gvr_deterministic_topk: true
attention_dp_config:
  enable_kv_balanced_admission: true
```

On the DEP16 servers, `max_batch_size` applies to each of the 16 attention-DP ranks.

#### Router

`router.yaml`:

```yaml
hostname: 0.0.0.0
port: 65120
backend: pytorch
gen_strip_message_history: true
gen_tokids_ctxbytes: true
server_keep_alive_timeout: 3600
num_workers: 4
context_servers:
  num_instances: 2
  router:
    type: conversation
  urls: ['<CTX0_HOST>:65100', '<CTX1_HOST>:65100']
generation_servers:
  num_instances: 1
  urls: ['<GEN0_HOST>:65110']
```

With one context server, set `context_servers.num_instances: 1` and `context_servers.urls: ['<CTX0_HOST>:65100']`.

The context and generation servers must be started with `--server_role context` and `--server_role generation`; the remaining Kimi K3.1 Pro settings are derived from the configuration and the server role.

### Launch the Servers

Each context and generation server is started through `trtllm-llmapi-launch`, one rank per GPU; its rank 0 serves HTTP. The example below starts a TEP8 context server and a TEP16 generation server with Slurm, inside one allocation of 6 trays in one rack (on Slurm clusters with topology segments, request the trays in one segment, for example `--segment=6`):

```bash
MODEL=<KIMI_K31_MODEL_DIR>
DRAFT=<DSPARK_DRAFT_MODEL_DIR>
REPO=/path/to/TensorRT-LLM                          # in-place build from the previous step
CFG=/path/to/configs   # tep8_ctx.yaml, tep16_gen.yaml and router.yaml (one context server)
IMAGE=/path/to/tensorrt-llm-container.sqsh
MOUNTS=${REPO}:${REPO},${MODEL}:${MODEL},${DRAFT}:${DRAFT},${CFG}:${CFG}:ro

start_server() {
    local role=$1 port=$2 name=$3; shift 3
    srun "$@" --ntasks-per-node 4 --gpus-per-node 4 --mpi=pmix \
        --container-image=${IMAGE} --container-mounts=${MOUNTS} \
        bash -c "
            ulimit -n 65536
            export UCX_CUDA_IPC_ENABLE_MNNVL=y
            export PATH=\"${REPO}/.venv-3.12/bin:\$PATH\"
            exec trtllm-llmapi-launch trtllm-serve ${MODEL} \
                --host 0.0.0.0 --port ${port} \
                --config ${CFG}/${name}.yaml \
                --server_role ${role} \
                --reasoning_parser kimi_k3 --tool_parser kimi_k3
        " &
}

start_server context    65100 tep8_ctx  -N 2 --ntasks 8  -w <CTX0_TRAYS>   # TEP8 on 2 trays
start_server generation 65110 tep16_gen -N 4 --ntasks 16 -w <GEN0_TRAYS>   # TEP16 on 4 trays
```

For the other deployments, start one server per entry of the router configuration: TEP8 context servers on 2 trays (8 ranks), the TEP32 generation server on 8 trays (32 ranks), and DEP16 servers on 4 trays (16 ranks) each. Then start the router on the rank-0 tray of the first context server:

```bash
srun -N 1 --ntasks 1 --overlap -w <CTX0_HOST> \
    --container-image=${IMAGE} --container-mounts=${MOUNTS} \
    bash -c "
        export PATH=\"${REPO}/.venv-3.12/bin:\$PATH\"
        exec trtllm-serve disaggregated -c ${CFG}/router.yaml \
            --server_start_timeout 7200 --request_timeout 7200
    "
```

> [!NOTE]
> `--server_role` (`context` or `generation`) is required on every disaggregated worker. `--reasoning_parser kimi_k3` and `--tool_parser kimi_k3` are the defaults for Kimi K3.1 Pro checkpoints when the options are omitted; they are shown for clarity, and `none` turns a parser off for clients that read only `content`. A command-line option overrides the same key in the YAML file.

Each server is ready when its rank 0 logs `Application startup complete`. The router waits for all servers, up to `--server_start_timeout` seconds, and then accepts requests on port 65120.

### Quick Start Without a Server (LLM API)

The script below runs the aggregated DEP16 configuration through the LLM API, with the DEP16 options of the configuration files minus the disaggregated-serving keys, and without speculative decoding. Save it as `quickstart.py` in a directory on the shared filesystem:

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (DeepSeekSparseAttentionConfig, KvCacheConfig,
                                 MoeConfig)


def main():
    with LLM(
            "<KIMI_K31_MODEL_DIR>",
            tensor_parallel_size=16,
            moe_expert_parallel_size=16,
            enable_attention_dp=True,
            trust_remote_code=True,
            max_batch_size=16,
            max_num_tokens=8192,
            max_seq_len=1048576,
            enable_chunked_prefill=True,
            kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True,
                                          dtype="nvfp4_ds_mla",
                                          tokens_per_block=64,
                                          free_gpu_memory_fraction=0.5,
                                          enable_block_reuse=False),
            moe_config=MoeConfig(max_num_tokens=32768),
            sparse_attention_config=DeepSeekSparseAttentionConfig(
                indexer_k_dtype="fp4",
                enable_heuristic_topk=True,
                gvr_deterministic_topk=True),
    ) as llm:
        outputs = llm.generate(["The capital of France is"],
                               SamplingParams(max_tokens=32, temperature=0))
        print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
```

Run it with one rank per GPU under `trtllm-llmapi-launch`, on 4 trays of one rack:

```bash
WORKDIR=/path/to/workdir   # the directory that holds quickstart.py

srun -N 4 --ntasks 16 --ntasks-per-node 4 --gpus-per-node 4 --mpi=pmix \
    --container-image=${IMAGE} \
    --container-mounts=${REPO}:${REPO},${MODEL}:${MODEL},${WORKDIR}:${WORKDIR} \
    bash -c "
        export PATH=\"${REPO}/.venv-3.12/bin:\$PATH\"
        exec trtllm-llmapi-launch python3 ${WORKDIR}/quickstart.py
    "
```

The same options in a YAML file passed to `trtllm-serve --config` start an aggregated DEP16 server; omit `--server_role` for it.

### Key YAML Options

These options are set in the YAML files passed to `trtllm-serve --config`. See the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full reference.

#### `tensor_parallel_size`, `moe_expert_parallel_size`, `moe_tensor_parallel_size`, `enable_attention_dp`

* **Description:** Select the server layout (TEP or DEP); see [Supported Configurations](#supported-configurations). The world size of each server equals `tensor_parallel_size`.

#### `trust_remote_code`

* **Description:** Must be `true`: the checkpoint's configuration and tokenizer code need it.

#### `kv_cache_config`

* **Options:**
  * `use_kv_cache_manager_v2`: `true` is required; the DSA indexer needs the KV cache manager V2.
  * `dtype`: `fp8` on the TEP servers and `nvfp4_ds_mla` on the DEP16 servers.
  * `tokens_per_block`: `64`.
  * `enable_block_reuse`, `block_reuse_config`: on the context servers, `policy: per_conversation` keeps the KV cache and KDA state of the last `max_num_turns` turns of each conversation (5 on TEP8, 3 on DEP16). The generation servers do not reuse blocks.
  * `mamba_state_config`: on the context servers, `additional_snapshot_offsets_from_end: [0]` snapshots the KDA state at the end of each prompt.
  * `avg_seq_len`: the average sequence length the KV cache manager V2 plans for; `150000` on the context servers.
  * `host_cache_size`: bytes of host memory per rank for offloaded blocks on the context servers.
  * `free_gpu_memory_fraction`: the fraction of free GPU memory given to the KV cache: `0.7` on the context servers, `0.8` on the TEP generation servers, `0.9` on the DEP16 generation server and `0.5` on an aggregated DEP16 server.

#### `sparse_attention_config`

* **Description:** `algorithm: dsa` with the heuristic top-k selection, made deterministic by `gvr_deterministic_topk`. `indexer_k_dtype: fp4` stores the DEP16 indexer keys in FP4; the TEP servers keep the FP8 default. `q_split_threshold: 1024` on the TEP8 context servers splits the indexer queries of prefill chunks above 1024 tokens across the ranks.

#### `speculative_config`

* **Description:** DSpark with `max_draft_len` equal to `block_size` and `speculative_model` set to the draft model. `context_kv_dtype: fp8` stores the draft's context KV in FP8. `context_kv_only: true` makes the context servers build that KV during prefill without drafting. `advanced_sampling_mode` on the generation servers selects the sampling kernels: `full` (TEP generation servers) applies `top_k`, `top_p` and `min_p`; `no_topk` (DEP16 generation server) skips `top_k`.

#### `prefill_cuda_graph_backend`, `prefill_capture_num_tokens`, `disable_mm_encoder`

* **Description:** On the TEP8 context servers, `prefill_cuda_graph_backend: breakable` captures prefill in CUDA graphs for the token counts in `prefill_capture_num_tokens`, together with `disable_mm_encoder: true`, which makes these servers text-only.

#### `cuda_graph_config`

* **Description:** The generation servers capture a CUDA graph for every entry of `batch_sizes`, with `enable_padding: true`. The DEP16 context server runs without CUDA graphs (`null`).

#### `moe_config`

* **Description:** `max_num_tokens` is the MoE chunk size: `33024` on the TEP8 context servers, `65536` on the DEP16 context servers and `32768` on an aggregated DEP16 server.

#### `disable_overlap_scheduler`

* **Description:** `true` on the context servers, `false` on the generation servers.

#### `attention_dp_config`

* **Description:** DEP16 only; sets how requests are assigned to the attention-DP ranks: `kv_cache_routing_conversation_affinity: true` and `kv_cache_routing_new_conv_placement: least_queued` on the context servers, `enable_kv_balanced_admission: true` on the generation server.

#### `cache_transceiver_config`

* **Description:** `backend: NIXL` with `transceiver_runtime: PYTHON`, `kv_cache_bounce_size_mb: 0` and `kv_transfer_timeout_ms: 600000`.

#### `num_serve_frontends`, `num_postprocess_workers`

* **Description:** `num_serve_frontends: 8` runs 8 HTTP front-end processes per server, and `num_postprocess_workers: 4` offloads detokenization on the generation servers.

### Kimi K3.1 Pro API Behavior

* **Reasoning:** Thinking is on by default. Turn it off per request with `"chat_template_kwargs": {"thinking": false}`, `"thinking": {"type": "disabled"}` or `"reasoning_effort": "none"`. The effort is set with `thinking.effort` or `reasoning_effort`: `low`, `medium`, `high` or `max`. Explicit `chat_template_kwargs` take precedence. The `kimi_k3` reasoning parser returns the reasoning in `reasoning_content` and the answer in `content`.
* **Tool calling:** The `kimi_k3` tool parser returns calls in `tool_calls`. `tool_choice` accepts `auto`, `required` and `none`; tools can also be declared on system messages (message-level tools, a Kimi API extension).
* **Strict tool calls (optional):** To constrain tool-call arguments to each tool's JSON schema, start every server (the context and generation servers, or the aggregated server) with `guided_decoding_backend: xgrammar` in its YAML file and `TRTLLM_KIMI_K3_STRICT_TOOL_GRAMMAR=1` in its environment, and send the tools with `"strict": true`. Tools without `strict` are not constrained, and the model can occasionally emit a tool call with missing arguments. The grammar constrains only the target model, so speculative-decoding acceptance can be somewhat lower.
* **Sampling:** Use `temperature: 1.0` and `top_p: 0.95`; set both in each request. On the DEP16 generation server (`advanced_sampling_mode: no_topk`), a request's `top_k` is not applied and `min_p` is not supported.
* **Usage reporting:** Streaming responses report `usage` in the final chunk without `stream_options`. `usage.prompt_tokens` excludes the 3-token generation-channel opener that ends the rendered prompt.
* **Images:** DEP16 (disaggregated and aggregated) accepts images as `image_url` content parts; the TEP configurations in this guide are text-only. Their TEP8 context servers run with `disable_mm_encoder: true` and reject image requests with HTTP 400.
* **Parsers:** Chat and agent clients use the default `kimi_k3` reasoning and tool parsers and read `reasoning_content` and `tool_calls`. For clients that read only `content` (for example AIPerf), start the servers with `--reasoning_parser none --tool_parser none`.
* **Client compatibility:** `prompt_cache_key`, which agent clients such as kimi-code send on every chat request, is accepted and ignored.
* **Conversation routing:** Requests that carry `"conversation_params": {"conversation_id": "<id>"}` are routed to the server that holds the conversation's cache. Without it, the router matches prompt prefixes.

## Testing API Endpoint

Send requests to the router, port `65120`, on the tray where it runs.

### Health Check

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://<ROUTER_HOST>:65120/health"
```

### Basic Chat

```shell
curl http://<ROUTER_HOST>:65120/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "<KIMI_K31_MODEL_DIR>",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "chat_template_kwargs": {"thinking": true},
    "temperature": 1, "top_p": 0.95,
    "max_tokens": 2048
}' -w "\n"
```

The answer is in `choices[0].message.content` and the reasoning in `choices[0].message.reasoning_content`. Add `"stream": true` for server-sent events.

### Reasoning

```shell
curl http://<ROUTER_HOST>:65120/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "<KIMI_K31_MODEL_DIR>",
    "messages": [{"role": "user", "content": "What is 17 * 23? Give the final number."}],
    "reasoning_effort": "low",
    "temperature": 1, "top_p": 0.95,
    "max_tokens": 4096
}' -w "\n"
```

`reasoning_effort` takes `low`, `medium`, `high` or `max`, and `none` answers without reasoning. Leave room in `max_tokens` for the reasoning.

### Tool Calling

```shell
curl http://<ROUTER_HOST>:65120/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "<KIMI_K31_MODEL_DIR>",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    }],
    "tool_choice": "auto",
    "temperature": 1, "top_p": 0.95,
    "max_tokens": 2048
}' -w "\n"
```

The call is returned in `choices[0].message.tool_calls` with `finish_reason` `tool_calls`. To continue, append the assistant message with its `tool_calls` and a `tool` message with the matching `tool_call_id`.

### Image Input (DEP16 only)

```shell
IMAGE_B64=$(base64 -w0 /path/to/image.png)

curl http://<ROUTER_HOST>:65120/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "<KIMI_K31_MODEL_DIR>",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"${IMAGE_B64}"'"}},
            {"type": "text", "text": "Describe this image in one sentence."}
        ]
    }],
    "temperature": 1, "top_p": 0.95,
    "max_tokens": 2048
}' -w "\n"
```

## Disaggregated Serving

* **KV transfer:** Each disaggregated deployment runs in one NVL72 rack, and the context and generation servers set `UCX_CUDA_IPC_ENABLE_MNNVL=y`, so the KV cache moves between the trays over NVLink.
* **Request flow:** The router sends each request to a context server, which runs the prefill, builds the DSpark draft's context KV and returns the first token. The generation server then pulls the KV cache over NIXL and decodes.
* **Routing:** `router.type: conversation` keeps the turns of a conversation on one context server, which reuses the KV cache and KDA state of the previous turns.
* **Router options:** `gen_strip_message_history` forwards only the last message to the generation server, which receives the prompt as token IDs; `gen_tokids_ctxbytes` relays those token IDs as one compact buffer. `num_workers: 4` runs four router processes on the public port.

See [Disaggregated Serving](../features/disagg-serving.md) for the general setup.

## Running Evaluations to Verify Accuracy (Optional)

Run the lm-evaluation-harness `gsm8k` task (5-shot) against the router:

```bash
lm_eval --model local-completions \
    --model_args "model=<KIMI_K31_MODEL_DIR>,base_url=http://<ROUTER_HOST>:65120/v1/completions,tokenizer=<KIMI_K31_MODEL_DIR>,tokenizer_backend=huggingface,trust_remote_code=True,tokenized_requests=False,num_concurrent=16,max_retries=3,timeout=7200,max_length=8192,max_gen_toks=256" \
    --tasks gsm8k --gen_kwargs max_gen_toks=256 --output_path ./kimi-k31-gsm8k
```

## Known Limitations

* **Aggregated serving does not reach 1M tokens per request.** An aggregated DEP16 server fits about 500k tokens per request in its KV cache; the disaggregated deployments serve the full 1,048,576.
* **Block reuse on an aggregated server shrinks its context.** With `enable_block_reuse: true`, the KDA recurrent states are kept in the KV cache pool, and their number grows with `max_batch_size`; at `free_gpu_memory_fraction: 0.5` they can lower `max_seq_len` to about 11.6k tokens. For prefix reuse on an aggregated server, use `max_batch_size: 8` with `kv_cache_config.mamba_state_config.additional_snapshot_offsets_from_end: [0]`; otherwise keep `enable_block_reuse: false`. The disaggregated deployments are not affected.
* **The TEP configurations in this guide are text-only.** Their TEP8 context servers run with `disable_mm_encoder: true` and reject image requests with HTTP 400. Images: DEP16 (disaggregated and aggregated).
* **Validated on GB300 NVL72 only**, with each disaggregated deployment in one rack.
* **Structured outputs** are not validated with these configurations, which set no `guided_decoding_backend`.
* **Sampling:** on the DEP16 generation server (`advanced_sampling_mode: no_topk`), `top_k` is ignored and `min_p` needs `full`.

## Troubleshooting Tips

| Symptom | Cause and fix |
| :-- | :-- |
| KV transfer errors, or a server allocates the wrong caches | Start every worker with `--server_role context` or `--server_role generation`. |
| Image requests fail on a TEP deployment | Expected: the TEP8 context servers run without the vision encoder. Use a DEP16 deployment (disaggregated or aggregated). |
| The router times out waiting for the servers | Raise `--server_start_timeout`. Check that each URL in `router.yaml` is the rank-0 tray of its server and that the ports match. |
| Slow KV transfer between servers | The trays are not in one NVL72 rack. Allocate the whole deployment in one rack. |
| Slow startup or slow first requests | TEP generation servers can take over an hour to become ready because of CUDA graph capture, and the first requests after a fresh install compile kernels once, so keep `FLASHINFER_WORKSPACE_BASE` on persistent node-local storage. |
| A client reports failed requests whose reply is all reasoning | The client reads only `content`. Use a reasoning-aware client, or start the servers with `--reasoning_parser none --tool_parser none`. |
| CUDA out-of-memory at startup | Lower `free_gpu_memory_fraction`; for host-memory exhaustion, lower `host_cache_size`. |
| CUDA out-of-memory at startup of an aggregated DEP16 server with `max_seq_len: 1048576` | Use `kv_cache_config.free_gpu_memory_fraction: 0.5` and `moe_config.max_num_tokens: 32768`; 0.7 does not fit. The KV cache then holds about 500k tokens per request, and longer prompts are rejected (HTTP 400); serve 1M-token prompts with the disaggregated deployments. |
| Stale file handle errors during startup | Point the JIT caches (`TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`) at node-local storage. |
| Model fails to load | Set `trust_remote_code: true`, keep `use_kv_cache_manager_v2: true`, and check that the checkpoint and the draft model are complete. |
| Ranks cannot reach each other | Check that the repository, checkpoint, draft model and configuration files resolve to the same paths on all trays, and that the rank count equals `tensor_parallel_size`. |
