<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment Guide for Kimi K3.1 Pro on TensorRT LLM - Blackwell (GB300)

## Introduction

This guide describes how to serve Kimi K3.1 Pro with TensorRT LLM on NVIDIA GB300 NVL72 systems. It covers the checkpoints, the supported deployments, building the software, aggregated and disaggregated serving with the DFlash2 drafter, the model-specific API behavior, example requests, accuracy verification and the known limitations. The configurations below are validated on GB300; configurations outside these tables are not validated.

Kimi K3.1 Pro is a hybrid Mixture-of-Experts (MoE) vision-language model, structured like Kimi K3:
* Most decoder layers use Kimi Delta Attention (KDA), a linear-attention mixer that keeps a constant-size recurrent state per sequence; the others use Multi-head Latent Attention (MLA).
* Every MLA layer adds a DeepSeek Sparse Attention (DSA) indexer, which selects the cached tokens each query attends to.
* The MoE block routes each token to 16 of 896 experts. The routed experts are stored in MXFP4; the attention, shared-expert and vision-tower weights are BF16.
* A MoonViT vision tower handles image inputs.

TensorRT LLM serves it as the architecture `KimiK31ForConditionalGeneration` (`model_type: kimi_k31`) through **modeling_v2** targets (`tensorrt_llm/_torch/modeling_v2/models/kimi_k31/`). A modeling_v2 target is one implementation for one combination of checkpoint, GPU architecture and parallel layout; routing picks it when the model is loaded. There is no other implementation of this architecture, so a deployment that no target claims fails to load.

## Prerequisites

* GPU: NVIDIA GB300 (`sm_103`) in an NVL72 rack, 4 GPUs per tray. Keep every server of a deployment, and for disaggregated serving every context and generation server, in one NVLink domain (one rack), so that the expert-parallel traffic and the KV cache transfer run over NVLink.
* Per-GPU memory decides which layouts fit:

  | Layout | Attention | Experts | GPUs | Per-rank weights |
  | :-- | :-- | :-- | --: | --: |
  | TEP8 | tensor-parallel (12 heads per rank) | EP8 | 8 | about 181 GiB |
  | DEP16 | data-parallel (replicated) | EP16 | 16 | about 184 GiB |

  Attention data parallelism keeps the BF16 attention, shared-expert and embedding weights whole on every rank, so DEP8 needs about 270 GiB per rank for weights alone and does not fit; use DEP16 for attention data parallelism.
* Multi-node launcher: Slurm with the pyxis/enroot container plugin and PMIx (`srun --mpi=pmix`), or an equivalent MPI launcher, to start one rank per GPU.
* A shared filesystem visible to all trays for the checkpoint, the drafter, the configuration files and the repository.
* Container runtime with NVIDIA GPU support, and CUDA driver 580 or later.

## Models

| Checkpoint | Weights | Use |
| :-- | :-- | :-- |
| [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) | Routed experts pre-quantized to MXFP4; BF16 attention, shared experts and vision tower | The model TensorRT LLM serves |
| [RadixArk/Kimi-K3.1-Dflash2-Preview](https://huggingface.co/RadixArk/Kimi-K3.1-Dflash2-Preview) | DFlash2 MLA drafter with a sliding window over its context | Speculative decoding (`speculative_config.speculative_model`) |

### Download the Checkpoint

Download both repositories once to the shared filesystem. Pass the directories to TensorRT LLM as they are; no conversion or quantization step is needed. The examples below use `/path/to/kimi-k3.1-pro` and `/path/to/kimi-k3.1-dflash2`; replace them with your local paths.

```bash
hf download moonshotai/Kimi-K3.1-Pro --local-dir /path/to/kimi-k3.1-pro
hf download RadixArk/Kimi-K3.1-Dflash2-Preview --local-dir /path/to/kimi-k3.1-dflash2
```

### Checkpoint Format

The checkpoint is in the Hugging Face format: `config.json` (`model_type: kimi_k31`, a composite config with a `kimi_linear` text config and a vision config), the weights, and the tokenizer, chat template and image processor together with their Python code. TensorRT LLM parses `config.json`, including the MXFP4 quantization of the routed experts, with its own config classes. The tokenizer and the image processor are loaded from the checkpoint, which needs `trust_remote_code: true`.

## Supported Configurations

| Deployment | GPUs | Servers | Targets | Validated |
| :-- | --: | :-- | :-- | :-- |
| Aggregated TEP8 | 8 | one server: `tensor_parallel_size: 8`, `moe_expert_parallel_size: 8` | `kimi_k31_pro_mxfp4__sm_103__tep` | GSM8K, GPQA Diamond, Kimi Vendor Verifier (including images) |
| Aggregated DEP16 | 16 | one server: `tensor_parallel_size: 16`, `moe_expert_parallel_size: 16`, `enable_attention_dp: true` | `kimi_k31_pro_mxfp4__sm_103__dep` | GSM8K, GPQA Diamond, Kimi Vendor Verifier (including images), BEAM (1M-token contexts) |
| Disaggregated: 2x TEP8 context, DEP16 generation, DCX | 32 | two TEP8 context servers, one DEP16 generation server, a router | context `__sm_103__tep`, generation `__sm_103__dep` | GSM8K, GPQA Diamond, Kimi Vendor Verifier (text), BEAM (1M-token contexts) |

All three run FP8 KV caches and DFlash2 speculative decoding.

Routing takes the attention-TP performance target for TEP8 with an FP8 KV cache, and the attention-DP performance target for DEP16. Every other admitted deployment, including the checkpoint's BF16 latents (`kv_cache_config.dtype: auto`), runs the general target `kimi_k31_pro_mxfp4__sm_10x__default`. With `modeling_v2: require`, a deployment no target accepts fails at startup with the first rule it breaks (see `routing.py` in the family directory).

| Feature | Status |
| :-- | :-- |
| DFlash2 speculative decoding | Supported: one-engine DFlash with the DFlash2 drafter and block verification (`use_rejection_sampling: true`). |
| Image input | Supported on the aggregated deployments. The disaggregated configuration below runs text-only context servers. |
| KV cache block reuse | Opt-in (`kv_cache_config.enable_block_reuse: true`); it also keeps KDA recurrent-state snapshots. Works with DFlash2. |
| Chunked prefill | Supported; required when prompts can be longer than `max_num_tokens`. |
| Disaggregated serving | Python transceiver over NIXL only; drafter context transfer (DCX) is always on with DFlash2. |
| Tool calling and reasoning output | Supported through the `kimi_k3` tool parser and the `kimi_k31` reasoning parser. |
| Structured outputs and grammar-enforced tool calls | Need `guided_decoding_backend: xgrammar`, which DFlash2 block verification (`use_rejection_sampling: true`) does not allow. See [Known Limitations](#known-limitations). |
| Pipeline parallelism, LoRA, beam search | Not supported or not validated. |

## Deployment Steps

### Install TensorRT LLM

Build from source in the TensorRT LLM development container (SBSA, aarch64 variant). From the repository root:

```bash
python3 scripts/build_wheel.py --use_ccache -a "103-real" --skip_building_wheel --install --yes
```

`--install` installs TensorRT LLM in editable mode into the virtual environment that `build_wheel.py` creates at the repository root (`.venv-3.12` in the current containers). The launch commands below put its `bin` directory first on `PATH`, so build with the repository at the same path the servers use. See [Building from Source](../installation/build-from-source.md) for other options.

### Configuration File

Create a YAML configuration file on the shared filesystem. The file below is the validated aggregated TEP8 configuration:

```shell
EXTRA_LLM_API_FILE=/path/to/kimi-k31.yml

cat << EOF > ${EXTRA_LLM_API_FILE}
modeling_v2: require
trust_remote_code: true
tensor_parallel_size: 8
moe_expert_parallel_size: 8
enable_attention_dp: false
max_batch_size: 8
max_num_tokens: 8192
max_seq_len: 131072
enable_chunked_prefill: true
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 3, 4, 5, 6, 7, 8]
moe_config:
  max_num_tokens: 32768
kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: fp8
  tokens_per_block: 64
  enable_block_reuse: true
  mamba_state_config:
    additional_snapshot_offsets_from_end: [0]
  free_gpu_memory_fraction: 0.5
sparse_attention_config:
  algorithm: dsa
  indexer_k_dtype: fp8
speculative_config:
  decoding_type: DFlash
  max_draft_len: 4
  speculative_model: /path/to/kimi-k3.1-dflash2
  use_rejection_sampling: true
EOF
```

For aggregated DEP16, change these keys:

| Key | DEP16 value |
| :-- | :-- |
| `tensor_parallel_size`, `moe_expert_parallel_size` | `16`, `16` |
| `enable_attention_dp` | `true`; `max_batch_size` and `cuda_graph_config` then apply to each rank |
| `max_seq_len` | `1048576` |
| `kv_cache_config.avg_seq_len` | `150000` |
| `attention_dp_config` | `kv_cache_routing_conversation_affinity: true`, `kv_cache_routing_new_conv_placement: least_tokens` |
| `stream_interval`, `num_serve_frontends`, `num_postprocess_workers` | `100`, `8`, `4` |

The GB300 DEP16 configuration with DFlash2 is available as **General deployment** in the [Recipe Selector](index.rst#recipe-selector).

### Launch the TensorRT LLM Server

The server is launched through the `trtllm-llmapi-launch` wrapper, one rank per GPU; rank 0 serves HTTP. The example starts the TEP8 server on 2 trays; for DEP16 use `-N 4 --ntasks 16` and the DEP16 file. On Slurm clusters with topology segments, request the trays in one segment (for example `--segment=4`).

```bash
MODEL=/path/to/kimi-k3.1-pro
DRAFTER=/path/to/kimi-k3.1-dflash2
REPO=/path/to/TensorRT-LLM   # the repository, built as in Install TensorRT LLM

srun -N 2 --ntasks 8 --ntasks-per-node 4 --mpi=pmix --gpus-per-node=4 \
    --container-image=/path/to/tensorrt-llm-devel-container.sqsh \
    --container-mounts=${REPO}:${REPO},${MODEL}:${MODEL},${DRAFTER}:${DRAFTER},${EXTRA_LLM_API_FILE}:${EXTRA_LLM_API_FILE}:ro \
    bash -c "
        ulimit -n 65536
        export PMIX_MCA_gds=hash
        # Node-local JIT caches: shared-filesystem defaults race across ranks.
        L=/tmp/kimi-k31-rank\${SLURM_PROCID:-0}
        export TRITON_CACHE_DIR=\$L/triton TORCHINDUCTOR_CACHE_DIR=\$L/inductor CUTE_DSL_CACHE_DIR=\$L/cute
        export TRTLLM_CACHE_DIR=\$L/trtllm XDG_CACHE_HOME=\$L/xdg FLASHINFER_WORKSPACE_BASE=\$L/flashinfer
        export PATH=\"${REPO}/.venv-3.12/bin:\$PATH\"
        exec trtllm-llmapi-launch trtllm-serve ${MODEL} \
            --host 0.0.0.0 --port 8000 \
            --config ${EXTRA_LLM_API_FILE} \
            --tool_parser kimi_k3 --reasoning_parser kimi_k31
    "
```

For DEP16, also export these in the `bash -c` command:

```bash
export TRTLLM_SERVER_DISABLE_GC=1 TRTLLM_WORKER_DISABLE_GC=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=1
export CUDA_SCALE_LAUNCH_QUEUES=4x        # more CUDA launch queues for the many concurrent streams
export TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1  # the default AllReduce strategy, without the warmup autotune
```

> [!NOTE]
> The number of ranks must equal `tensor_parallel_size`, and the repository, checkpoint, drafter and configuration file must resolve to the same paths on all trays. `--reasoning_parser auto` selects `kimi_k31` from the checkpoint. A command-line option that is also set in the YAML file overrides the YAML value.

The server is ready when rank 0 logs `Application startup complete`.

### Quick Start Without a Server (LLM API)

The script below runs the TEP8 configuration through the LLM API. Save it as `quickstart.py` in a directory on the shared filesystem:

```python
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (CudaGraphConfig, DeepSeekSparseAttentionConfig,
                                 DFlashDecodingConfig, KvCacheConfig, MoeConfig)


def main():
    with LLM(
            "/path/to/kimi-k3.1-pro",
            modeling_v2="require",
            trust_remote_code=True,
            tensor_parallel_size=8,
            moe_expert_parallel_size=8,
            max_batch_size=8,
            max_num_tokens=8192,
            max_seq_len=131072,
            enable_chunked_prefill=True,
            cuda_graph_config=CudaGraphConfig(enable_padding=True,
                                              max_batch_size=8),
            moe_config=MoeConfig(max_num_tokens=32768),
            kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True,
                                          dtype="fp8",
                                          tokens_per_block=64,
                                          free_gpu_memory_fraction=0.5),
            sparse_attention_config=DeepSeekSparseAttentionConfig(
                indexer_k_dtype="fp8"),
            speculative_config=DFlashDecodingConfig(
                max_draft_len=4,
                speculative_model="/path/to/kimi-k3.1-dflash2",
                use_rejection_sampling=True),
    ) as llm:
        outputs = llm.generate(["The capital of France is"],
                               SamplingParams(max_tokens=32, temperature=0))
        print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
```

Run it with the server's `srun` command, mounting the directory that holds the script and replacing the `trtllm-serve ...` line with `exec trtllm-llmapi-launch python3 /path/to/workdir/quickstart.py`: one rank per GPU under `trtllm-llmapi-launch`.

### Key YAML Options

These options are set in the YAML file passed to `trtllm-serve --config`. See the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full reference.

#### `modeling_v2`

* **Description:** Set it to `require`, so that a configuration no Kimi K3.1 target serves fails at startup with the reason. `auto` (the default) selects the same targets, but reports a refused configuration only as "no target". `trust_remote_code: true` is also required: the checkpoint's tokenizer and image processor are Python code in the checkpoint.

#### `tensor_parallel_size`, `moe_expert_parallel_size`, `moe_tensor_parallel_size`, `enable_attention_dp`

* **Description:** Select the layout; see [Supported Configurations](#supported-configurations). The world size equals `tensor_parallel_size`; pipeline parallelism is not supported. With attention DP, use 16 ranks.

#### `kv_cache_config`

* **Options:**
  * `use_kv_cache_manager_v2`: `true` is required.
  * `dtype`: `fp8` (FP8 latent cache, the performance targets) or `auto` (the checkpoint's BF16 latents, the general target). Other values are refused.
  * `tokens_per_block`: `64`, the model default.
  * `enable_block_reuse`, `block_reuse_config`: block reuse is off by default. With it on, the cache also keeps KDA recurrent-state snapshots. `block_reuse_config.policy: per_conversation` with `max_num_turns` keeps the last turns of each conversation on a context server.
  * `mamba_state_config`: where the KDA state is snapshotted for reuse. `additional_snapshot_offsets_from_end: [0]` takes one at each prompt end, which multi-turn traffic reuses. For many questions over one long shared prefix, add a snapshot inside the shared history, for example `[0, 1024]` with `block_reuse_config.enable_branch_snapshot: true`.
  * `avg_seq_len`: the average sequence length the KV cache manager plans for.
  * `free_gpu_memory_fraction`: the share of free GPU memory for the KV cache.

#### `sparse_attention_config`

* **Description:** `algorithm: dsa` with FP8 indexer keys (`indexer_k_dtype: fp8`). The indexer always runs an exact Top-K in a fixed order (the deterministic GVR on GB300), so the Top-K options `enable_heuristic_topk` and `gvr_deterministic_topk` have no effect. `q_split_threshold: 1024` on the context servers splits the indexer queries of long prefill chunks across the ranks. The DSA sizes come from the checkpoint and the target's model defaults.

#### `speculative_config`

* **Description:** `decoding_type: DFlash` with the DFlash2 drafter as `speculative_model` and `use_rejection_sampling: true` (block verification). The verify width is `max_draft_len + 1`. `advanced_sampling_mode: no_topk` skips the sampler's top-k filter; top-p still applies.

#### `attention_dp_config`

* **Description:** DEP16 only. `kv_cache_routing_conversation_affinity: true` pins the requests that carry the same `conversation_params.conversation_id` (or, without one, the same `X-Session-ID` header) to the attention-DP rank of the conversation's first turn, so its cached prefix is reused there. `kv_cache_routing_new_conv_placement: least_tokens` places a new conversation on the rank with the fewest active prompt tokens. `enable_kv_balanced_admission: true` on a disaggregated generation server admits transferred requests by per-rank KV load.

#### `cache_transceiver_config`

* **Description:** Disaggregated serving only: `backend: NIXL` with `transceiver_runtime: PYTHON`, the only transceiver that moves the KDA recurrent state. See [Disaggregated Serving](#disaggregated-serving).

### Kimi K3.1 Pro API Behavior

* **Reasoning:** Thinking is on by default, with effort `max`. Turn it off per request with `"thinking": {"type": "disabled"}`, `"reasoning_effort": "none"` or `"chat_template_kwargs": {"thinking": false}`. Set the effort with `thinking.effort` or `reasoning_effort`: `low`, `high` or `max`; a `thinking.effort` outside these values returns HTTP 400, and a `reasoning_effort` with no Kimi equivalent (for example `medium`) leaves the default. Explicit `chat_template_kwargs` take precedence, then the `thinking` object, then `reasoning_effort`. With `--reasoning_parser kimi_k31`, the reasoning is returned in `reasoning_content` and the answer in `content`.
* **Tool calling:** With `--tool_parser kimi_k3`, calls are returned in `tool_calls`. `tool_choice` accepts `auto`, `required` and `none`. Tools can also be declared on system messages (message-level tools, a Kimi API extension). With DFlash2, `tool_choice: "required"` is requested through the chat template: the model normally calls a tool, but a response without one is possible.
* **Sampling:** Use `temperature` 1.0 and `top_p` 0.95. `TRTLLM_KIMI_PARAM_POLICY=1` in the server environment enforces Kimi's pinned sampling parameters: `top_p` 0.95 (unset or `1.0` is coerced, other values return HTTP 400), no presence or frequency penalty, `n` 1 and `temperature` within [0, 1]. Kimi Vendor Verifier certification runs need it.
* **Images:** The aggregated deployments accept images as `image_url` content parts (data URLs, or `http(s)` URLs that the server downloads). Image tokens count against `max_seq_len`.
* **Usage reporting:** Streaming responses report `usage` in the final chunk without `stream_options`. `usage.prompt_tokens` excludes the 3-token generation-channel opener that ends the rendered prompt.
* **Conversation routing:** Requests that carry `"conversation_params": {"conversation_id": "<id>"}` stay on the attention-DP rank (DEP16) or the context server (disaggregated) that holds the conversation's cache.

## Testing API Endpoint

The server (for disaggregated serving, the router) listens on port `8000` of its rank-0 tray. Send requests to that tray's hostname; `localhost` only works on the tray itself.

### Health Check

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

### Basic Chat

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/kimi-k3.1-pro",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 1, "top_p": 0.95,
    "max_tokens": 2048
}' -w "\n"
```

The answer is in `choices[0].message.content` and the reasoning in `choices[0].message.reasoning_content`. Add `"stream": true` for server-sent events.

### Reasoning

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/kimi-k3.1-pro",
    "messages": [{"role": "user", "content": "What is 17 * 23? Give the final number."}],
    "reasoning_effort": "low",
    "temperature": 1, "top_p": 0.95,
    "max_tokens": 4096
}' -w "\n"
```

`reasoning_effort` takes `low`, `high` or `max` (the default), and `none` answers without reasoning. Leave room in `max_tokens` for the reasoning.

### Tool Calling

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/kimi-k3.1-pro",
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

### Image Input

```shell
IMAGE_B64=$(base64 -w0 /path/to/image.png)

curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/kimi-k3.1-pro",
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

Disaggregated serving runs separate context (prefill) and generation (decode) servers behind a router (`trtllm-serve disaggregated`). The router sends each request to a context server, which runs the prefill and returns the first token; the generation server then pulls the KV cache over NIXL and decodes. See [Disaggregated Serving](../features/disagg-serving.md) for the general setup.

**Settings common to the context and generation servers:**

* The Python transceiver over NIXL on every context and generation server, with the same `kv_transfer_timeout_ms` on both sides. Do not set `max_tokens_in_buffer`: with the Python transceiver it bounds the transfer window, and sends that wait for a generation peer can then stall the context server.

  ```yaml
  cache_transceiver_config:
    backend: NIXL
    transceiver_runtime: PYTHON
    kv_cache_bounce_size_mb: 0
    kv_transfer_timeout_ms: 60000
  ```

* An FP8 KV cache and KV cache manager V2 on both sides. Block reuse is on for the context servers (multi-turn prefixes) and off for the generation servers.
* The context servers run without the overlap scheduler and without the vision tower (`disable_mm_encoder: true`), so the configuration below is text-only. For image requests, use a context server with the vision tower: remove `disable_mm_encoder`, `prefill_cuda_graph_backend` and `prefill_capture_num_tokens` from its configuration and leave `TLLM_MULTIMODAL_DISAGGREGATED` unset.
* **Drafter context transfer (DCX)** is always on with DFlash2: the KV cache transfer carries the drafter's prompt context, so the generation server drafts from the first token. A context server (`--server_role context`) runs a context-only drafter. Set its `max_draft_len` to the generation server's. A pair in which only one side carries the drafter context is refused at registration.
* Environment on every context and generation rank, in addition to the DEP16 environment of [Launch the TensorRT LLM Server](#launch-the-tensorrt-llm-server):

  ```bash
  unset UCX_TLS                       # a container-default UCX_TLS=tcp breaks the NIXL setup
  export UCX_CUDA_IPC_ENABLE_MNNVL=y  # KV transfer over NVLink across trays
  export TRTLLM_KV_TRANSFER_NUM_THREADS=8
  # Context servers only:
  export TLLM_V2_CTX_DEMOTE_UNCACHED_TOKENS=16384 TLLM_V2_CTX_DEMOTE_MAX_ITERS=100  # short cached turns go ahead of long uncached prefills
  export TLLM_MULTIMODAL_DISAGGREGATED=1                                             # text-only context server
  # Generation servers only: the DSA indexer's logits workspace, which the KV cache budget reserves.
  export TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET=536870912
  ```

### Context Server (TEP8)

```yaml
modeling_v2: require
trust_remote_code: true
tensor_parallel_size: 8
moe_expert_parallel_size: 8
enable_attention_dp: false
disable_overlap_scheduler: true
enable_chunked_prefill: true
disable_mm_encoder: true
max_batch_size: 8
max_num_tokens: 16384
max_seq_len: 1048576
stream_interval: 10
num_serve_frontends: 8
cuda_graph_config:
  max_batch_size: 1
prefill_cuda_graph_backend: breakable
prefill_capture_num_tokens: [256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096]
kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: fp8
  tokens_per_block: 64
  enable_block_reuse: true
  block_reuse_config:
    policy: per_conversation
    max_num_turns: 4
  mamba_state_config:
    # KDA snapshots at each prompt end and 16K to 36K tokens into the prompt (shared system prompts).
    additional_snapshot_offsets_from_end: [0]
    additional_snapshot_offsets_from_start: [16384, 20480, 24576, 28672, 32768, 36864]
  avg_seq_len: 150000
  free_gpu_memory_fraction: 0.65
moe_config:
  max_num_tokens: 33024
sparse_attention_config:
  algorithm: dsa
  q_split_threshold: 1024
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  kv_cache_bounce_size_mb: 0
  kv_transfer_timeout_ms: 60000
speculative_config:
  decoding_type: DFlash
  max_draft_len: 5
  speculative_model: /path/to/kimi-k3.1-dflash2
  use_rejection_sampling: true
```

### Generation Server (DEP16)

```yaml
modeling_v2: require
trust_remote_code: true
tensor_parallel_size: 16
moe_expert_parallel_size: 16
enable_attention_dp: true
disable_overlap_scheduler: false
max_batch_size: 32
max_num_tokens: 256
max_seq_len: 1048576
stream_interval: 100
num_serve_frontends: 8
num_postprocess_workers: 4
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
kv_cache_config:
  use_kv_cache_manager_v2: true
  dtype: fp8
  tokens_per_block: 64
  enable_block_reuse: false
  avg_seq_len: 140000
  free_gpu_memory_fraction: 0.9
speculative_config:
  decoding_type: DFlash
  max_draft_len: 5
  speculative_model: /path/to/kimi-k3.1-dflash2
  advanced_sampling_mode: no_topk
  use_rejection_sampling: true
attention_dp_config:
  enable_kv_balanced_admission: true
sparse_attention_config:
  algorithm: dsa
cache_transceiver_config:
  backend: NIXL
  transceiver_runtime: PYTHON
  kv_cache_bounce_size_mb: 0
  kv_transfer_timeout_ms: 60000
```

### Router and Launch

The router lists the context and generation servers and routes the context servers by conversation, so a conversation's turns reach the context server that holds its prefix. Service discovery (`disagg_cluster`) backed by etcd lets the router drop a server that stops sending heartbeats, and refuse requests while fewer than `minimal_instances` servers are active. Start an etcd server next to the router (the TensorRT LLM container ships `etcd`) and give the router and every context and generation server the same `disagg_cluster` block:

```bash
etcd --name kimi-k31 --data-dir /tmp/etcd-kimi-k31 \
    --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://<router_host>:2379 &
```

`router.yaml`:

```yaml
hostname: 0.0.0.0
port: 8000
backend: pytorch
gen_strip_message_history: true
gen_tokids_ctxbytes: true
server_keep_alive_timeout: 3600
num_workers: 4
context_servers:
  num_instances: 2
  router:
    type: conversation
  urls:
    - "<ctx0_host>:8001"
    - "<ctx1_host>:8001"
generation_servers:
  num_instances: 1
  router:
    type: load_balancing
  urls:
    - "<gen_host>:8002"
disagg_cluster:
  cluster_uri: etcd://<router_host>:2379
  cluster_name: kimi-k31
  minimal_instances:
    context_servers: 1
    generation_servers: 1
  heartbeat_interval_sec: 1
  inactive_timeout_sec: 3
```

`gen_strip_message_history` forwards only the last message to the generation server, which receives the prompt as token IDs, and `gen_tokids_ctxbytes` relays those IDs as one compact buffer. Launch each server as in [Launch the TensorRT LLM Server](#launch-the-tensorrt-llm-server), with its configuration file, the environment above and its role, then the router:

```bash
trtllm-serve ${MODEL} --host 0.0.0.0 --port 8001 --config ctx.yaml --server_role context \
    --tool_parser kimi_k3 --reasoning_parser kimi_k31
trtllm-serve ${MODEL} --host 0.0.0.0 --port 8002 --config gen.yaml --server_role generation \
    --tool_parser kimi_k3 --reasoning_parser kimi_k31
trtllm-serve disaggregated -c router.yaml --server_start_timeout 7200 --request_timeout 7200
```

## Running Evaluations to Verify Accuracy (Optional)

The accuracy tests are in `tests/integration/defs/accuracy/modeling_v2/kimi_k31/`, one class per target, parameterized by deployment. Set `LLM_MODELS_ROOT` to a directory holding `Kimi-K3.1-Pro` and `Kimi-K3.1-Dflash2-Preview`. TEP8 and DEP16 span 2 and 4 trays; run them as the multi-node CI stages do, one MPI rank per GPU (the entries are in `tests/integration/test_lists/test-db/l0_gb300_multi_nodes_node2_gpu8.yml` and `l0_gb300_multi_nodes_node4_gpu16.yml`). For example, GSM8K on the attention-TP performance target at TEP8:

```bash
cd tests/integration/defs
LLM_MODELS_ROOT=/path/to/models pytest \
    "accuracy/modeling_v2/kimi_k31/kimi_k31_pro_mxfp4__sm_103__tep/test_accuracy.py::TestModelingV2KimiK31ProMxfp4Sm103Tep::test_gsm8k[tep8]"
```

| Test | Deployment | Reference |
| :-- | :-- | :-- |
| `test_gsm8k` | TEP8 / DEP16, FP8 KV cache | GSM8K 97.94 (`references/gsm8k.yaml`) |
| `test_gpqa_diamond` | DEP16, FP8 KV cache | GPQA Diamond 91.41 (`references/gpqa_diamond.yaml`) |
| `test_gsm8k_dflash2` | TEP8 / DEP16, DFlash2 at verify width 5 | GSM8K 97.94 on both layouts. TEP8 also checks a minimum acceptance length of 4.09 (`references/acceptance_length.yaml`) |

## Known Limitations

* **Guided decoding and DFlash2.** DFlash2 with block verification does not allow a guided-decoding backend, so `response_format` is not constrained and `tool_choice: "required"` is requested through the chat template on these deployments. Adding `guided_decoding_backend` to a configuration above fails at startup (`use_rejection_sampling=True is not supported for this configuration: guided decoding is enabled`). For grammar-enforced tool calls and structured outputs, serve without speculative decoding with `guided_decoding_backend: xgrammar` and `--reasoning_parser kimi_k31` on an aggregated server; the guided-decoding tests were validated this way. Keeping DFlash2 with `use_rejection_sampling: false` is also accepted: sampled requests are then verified by exact match, which accepts fewer drafted tokens per step than block verification, so decoding is slower at the recommended `temperature` of 1.0 (greedy requests are unaffected).
* **Images in disaggregated serving.** The disaggregated configuration above runs text-only context servers.
* **Long contexts.** The aggregated TEP8 configuration is validated at `max_seq_len` 131072; use DEP16 or the disaggregated configuration for contexts up to 1,048,576 tokens.
* **KV cache dtypes.** `fp8` and `auto` only.
* **Determinism.** Greedy decoding is not bitwise reproducible across runs and batch compositions; scores vary by a few tenths of a point between runs.

## Troubleshooting Tips

| Symptom | Cause and fix |
| :-- | :-- |
| Startup fails with a modeling_v2 refusal | With `modeling_v2: require`, the error names the first rule the configuration breaks, for example a layout the heads or experts do not divide, an unsupported KV cache dtype or a non-NIXL transceiver. Compare the configuration with [Supported Configurations](#supported-configurations). |
| CUDA out-of-memory at load | Attention DP needs 16 ranks. Otherwise lower `free_gpu_memory_fraction`, `max_batch_size` or `max_seq_len`; the KDA layers keep per-request FP32 recurrent state outside the paged KV pool. |
| Startup is refused with the DFlash context-buffer budget message | Lower `max_seq_len` or `max_batch_size`, or move long-context serving to DEP16. Do not disable the check. |
| A disaggregated peer is refused at registration because the drafter context is on one side only | Run DFlash2 with the same drafter and `use_kv_cache_manager_v2: true` on the context and the generation servers. |
| NIXL setup fails or the transceiver hangs in setup | Unset a container-default `UCX_TLS=tcp` on every rank. |
| The context server stops scheduling while its KV cache is full of finished sends | Remove `cache_transceiver_config.max_tokens_in_buffer`, and use the same `kv_transfer_timeout_ms` on both sides. |
| The router keeps sending requests to a server that died | Use `disagg_cluster` service discovery with etcd (see [Router and Launch](#router-and-launch)). |
| A client reports replies that are all reasoning | The client reads only `content`. Use a reasoning-aware client, or start the server without `--reasoning_parser`. |
| Stale file handle or cache errors during startup | Point the JIT caches (`TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE` and the others in the launch command) at node-local storage. |
| Ranks cannot reach each other | Check that the image, checkpoint, drafter, configuration files and repository resolve to the same paths on all trays, and that the rank count equals `tensor_parallel_size`. |
