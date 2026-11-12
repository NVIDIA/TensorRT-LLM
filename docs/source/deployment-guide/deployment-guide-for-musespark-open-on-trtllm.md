<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deployment Guide for Muse Spark Open on TensorRT LLM - Blackwell

## Introduction

This deployment guide provides step-by-step instructions for running the Muse Spark Open (816B-A42B) model using TensorRT LLM on NVIDIA Blackwell GPUs. The deployment configuration in this guide was validated on NVIDIA GB300 NVL GPUs. It covers preparing the model weights, configuring TensorRT LLM parameters, launching the server on a single 4-GPU node, and validating inference output.

Muse Spark Open is a large multimodal sparse Mixture-of-Experts (MoE) model. Its 62 decoder layers are all sparse: each MoE block routes every token to 8 of 256 experts, with no shared expert. Attention interleaves two window regimes in a repeating 3:1 pattern — three sliding-window layers (2,048-token window) followed by one full-attention layer (46 sliding-window and 16 full-attention layers in total) — with 128 query heads of head dimension 64 grouped over 16 KV heads. The model applies final logit softcapping (20.0), uses a 202,048-token vocabulary, and supports context lengths up to 1M tokens (`max_position_embeddings: 1048576`). The published checkpoint quantizes the routed-expert weights to NVFP4 (group size 16, via ModelOpt); the attention projections, the LM head, and the first and last decoder layers stay in BF16.

Beyond text, the checkpoint carries a vision tower (image and video input) and a convolutional audio tower (audio input from a 24 kHz feature extractor). In TensorRT LLM the checkpoint (`model_type: muse_spark`, architecture `MuseSparkForConditionalGeneration`) is loaded through the MuseSpark model path.

The configuration walkthrough focuses on the validated aggregated serving deployment: **TP4/EP4 on a single 4-GPU node** (`tensor_parallel_size: 4`, `moe_expert_parallel_size: 4`), with an FP8 KV cache, CUDA graphs, chunked prefill, and KV-cache block reuse enabled.

## Prerequisites

* GPU: 4x NVIDIA Blackwell GPUs (`SM100` family) on one node. The configuration in this guide was validated on GB300 NVL GPUs (`SM103`).
* OS: Linux
* Drivers: CUDA Driver 580 or later
* Container runtime with NVIDIA GPU support
* Disk/shared filesystem space for the checkpoint and the configuration file

## Models

* A complete Hugging Face-format Muse Spark Open checkpoint and tokenizer, e.g. [`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) (`model_type: muse_spark`, architecture `MuseSparkForConditionalGeneration`). The checkpoint ships the routed-expert weights pre-quantized to NVFP4; no additional quantization step is needed.

The examples below use `/path/to/musespark-open-checkpoint` — replace it with your local path or the Hub id. The Hub repository is gated: request access on its Hugging Face page and set `HF_TOKEN` before serving it by id.

## Feature Support Notes

* **Blackwell (`SM100` family) GPUs are supported.** The results and constraints in this guide were validated on GB300 NVL (`SM103`).
* **Aggregated serving with TP4/EP4 is the validated deployment.** Attention runs tensor-parallel (`enable_attention_dp: false`) and the 256 experts shard 4-way (`moe_expert_parallel_size: 4`).
* **The model runs on the KV cache manager v2**, the default; setting `kv_cache_config.use_kv_cache_manager_v2: false` is refused.
* **FP8 KV cache is required with chunked prefill or KV reuse on `SM103`.** For head dimension 64 on `SM103`, the fused paged-context FMHA kernel exists only for the FP8 KV-cache dtype. Paged-context FMHA is exercised whenever a request enters attention with a non-zero cached prefix — chunked prefill, KV-cache block reuse, or speculative draft verification — so those features require `kv_cache_config.dtype: fp8` on `SM103`; a 16-bit KV cache in that combination is refused rather than risking silent accuracy loss. The recommended configuration below satisfies this.
* **CUDA graphs and the overlap scheduler are enabled** (`disable_overlap_scheduler: false`, `cuda_graph_config.max_batch_size: 256` with `enable_padding: true`).
* **Chunked prefill is supported and enabled** (`enable_chunked_prefill: true`).
* **KV-cache block reuse is supported and enabled** (`enable_block_reuse: true`, `enable_partial_reuse: true`).
* **Guided decoding is supported** with the `xgrammar` backend (`guided_decoding_backend: xgrammar`), including JSON-schema constrained output.
* **Speculative decoding is supported** with a DFlash2 drafter and with model-free suffix-automaton (SA) drafting; see [Speculative Decoding](#speculative-decoding) below.
* **Disaggregated serving:** the KV cache manager v2 keeps disaggregated serving on the NIXL backend with the Python transceiver runtime, which are the defaults (`cache_transceiver_config: {backend: NIXL, transceiver_runtime: PYTHON}`); other transceiver configurations are refused at startup. Some disaggregated configurations additionally require adjusting the attention-kernel library selection with `TLLM_FMHA_LIBS=-fallback` in the server environment (this removes the fallback attention library from the default selection order). On GB300 (`SM103`, aarch64) also set `UCX_TLS=cuda_copy,cuda_ipc,sm,self,tcp` in every worker environment: with the default UCX transport list, NIXL memory registration fails at startup (`registerMem: registration failed for the specified or all potential backends`).

## Deployment Steps

### Run Docker Container

Run the docker container using the TensorRT LLM NVIDIA NGC image. The image is multi-arch (x86_64 and aarch64), so it runs on GB300 trays directly:

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
* See the <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags> for all the available containers. The containers published in the main branch weekly have `rcN` suffix, while the monthly release with QA tests has no `rcN` suffix. Use a release that includes Muse Spark Open support; the `rc` releases have the latest model and feature support.

If you want to use latest main branch, you can choose to build from source to install TensorRT LLM, the steps refer to [https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html](https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html)

> **All commands below should be run inside the Docker container.**

### Recommended Performance Settings

**Treat these as a starting point and tune the parameters for your workload.**

Create a YAML configuration file. The settings below are the validated TP4/EP4 aggregated serving configuration, matching `examples/musespark_open/serve_config.yaml`:

```shell
EXTRA_LLM_API_FILE=/path/to/musespark-open-config.yml

cat << EOF > ${EXTRA_LLM_API_FILE}
# Muse Spark Open aggregated serving: attention tensor-parallel across 4
# GPUs, 256 experts sharded 4-way.
tensor_parallel_size: 4
moe_expert_parallel_size: 4
enable_attention_dp: false
max_batch_size: 256
max_num_tokens: 8192
max_seq_len: 49152
trust_remote_code: true
enable_chunked_prefill: true
disable_overlap_scheduler: false
attn_backend: TRTLLM
allreduce_strategy: NCCL
guided_decoding_backend: xgrammar
cuda_graph_config:
  max_batch_size: 256
  enable_padding: true
moe_config:
  backend: TRTLLM
kv_cache_config:
  dtype: fp8
  enable_block_reuse: true
  enable_partial_reuse: true
  free_gpu_memory_fraction: 0.8
EOF
```

Notes:

* `kv_cache_config.dtype: fp8` is required on `SM103` whenever chunked prefill or block reuse is enabled (see Feature Support Notes).
* `max_seq_len: 49152` is the validated serving window; the model itself supports up to 1M tokens of context. Longer windows need proportionally more KV-cache memory.

### Launch the TensorRT LLM Server

Muse Spark Open at TP4/EP4 fits on a single 4-GPU node, so `trtllm-serve` can be launched directly (no multi-node launcher is needed):

```bash
MODEL=/path/to/musespark-open-checkpoint

trtllm-serve ${MODEL} \
    --host 0.0.0.0 --port 8000 \
    --config ${EXTRA_LLM_API_FILE} \
    --tool_parser musespark \
    --reasoning_parser musespark
```

The `--tool_parser` / `--reasoning_parser` flags enable tool-call and reasoning-content extraction on `/v1/chat/completions`. `auto` also selects `musespark` for both when `${MODEL}` is a local checkpoint directory, because the resolution reads the checkpoint's `config.json`; when serving by Hub id, pass `musespark` explicitly.

The tokenizer and chat template are used as shipped in the checkpoint; no custom tokenizer is needed.

TensorRT LLM will load weights and select kernels during startup. The server is successfully launched when the following log is shown:

```text
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

### Key YAML Options

These options are set within the YAML file passed to `trtllm-serve` via the `--config` argument. Only the MuseSpark-relevant knobs are described here; see the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full reference.

#### `tensor_parallel_size` / `moe_expert_parallel_size`

* **Description:** The validated deployment uses `4` for both: attention is tensor-parallel across the 4 GPUs and the 256 routed experts shard 4-way (64 experts per rank).

#### `kv_cache_config`

* **Options:**
  * `dtype`: `fp8`. Required on `SM103` when chunked prefill or block reuse is enabled (head-dimension-64 paged-context constraint; see Feature Support Notes).
  * `enable_block_reuse` / `enable_partial_reuse`: `true` in the validated configuration; repeated prefixes across requests then reuse cached KV blocks.
  * `free_gpu_memory_fraction`: `0.8` in the validated configuration. Lower it if you hit out-of-memory errors.

#### `cuda_graph_config`

* **Description:** The validated configuration captures CUDA graphs up to `max_batch_size: 256` with `enable_padding: true`, which pads generation batches to a captured size.

#### `enable_chunked_prefill`

* **Description:** `true` in the validated configuration; prompts longer than `max_num_tokens` are scheduled across multiple prefill steps.

#### `guided_decoding_backend`

* **Description:** Set to `xgrammar` to enable guided decoding (JSON mode, JSON schema, structural tags) on the OpenAI-compatible endpoints.

#### `attn_backend` / `allreduce_strategy` / `moe_config.backend`

* **Description:** The validated configuration uses the `TRTLLM` attention backend, `NCCL` all-reduce, and the `TRTLLM` MoE backend.
* **`allreduce_strategy`:** For this model, `AUTO` (the default) is replaced by `NCCL`, `NCCL_SYMMETRIC` is rejected, and any other value is used as given.

#### `trust_remote_code`

* **Description:** `true` in the validated configuration.

## Multimodal Inputs

Muse Spark Open accepts image, video, and audio inputs alongside text:

* **Image and video** are encoded by the checkpoint's vision tower (14x14 patches, 2x2 spatial merge, 2-frame temporal patching). Per the checkpoint's processor configuration, an image contributes up to 4,096 tokens and a video contributes up to 256 tokens per sampled frame (2 frames per second, up to 600 frames).
* **Audio** is encoded by the convolutional audio tower from a 24 kHz feature extractor; each audio token covers 1,920 input samples (80 ms).

Token budgets for multimodal items count against `max_num_tokens` and `max_seq_len` like text tokens; size long-video requests accordingly. For text-only serving, `disable_mm_encoder: true` skips loading the vision and audio towers, which leaves more GPU memory for the KV cache; image, video, and audio requests are then rejected.

## Speculative Decoding

Muse Spark Open supports speculative decoding with a **DFlash2 drafter** — a small draft model that cross-attends to hidden states captured from the target model and predicts multiple draft tokens in one forward pass. The published drafter checkpoint is [`nvidia/Muse-Spark-Open-816B-A42B-DFlash2`](https://huggingface.co/nvidia/Muse-Spark-Open-816B-A42B-DFlash2). `examples/musespark_open/spec_dflash.yaml` holds a typical configuration, which drafts `K = 7` tokens per step:

```yaml
speculative_config:
  decoding_type: DFlash
  max_draft_len: 7
  speculative_model: nvidia/Muse-Spark-Open-816B-A42B-DFlash2
```

The drafter keeps a context buffer for every batch slot, sized for `max_seq_len`. With the recommended configuration (`max_batch_size: 256`, `max_seq_len: 49152`) it takes about 60 GiB per GPU, which comes out of the KV cache: on one GB300 node the KV cache shrinks from about 108 GiB to about 50 GiB per GPU. Lowering `max_batch_size` or `max_seq_len` shrinks the buffer proportionally. With this configuration the drafter measured a mean acceptance length of 4.25 tokens per step on coding prompts at temperature 0.

DFlash requires the `TRTLLM` attention backend for the target model — the backend the validated aggregated recipe above already ships; a FlashInfer-attention configuration is refused at startup.

Draft-token acceptance is strict: an accepted token is exactly the token the target model emits at that position. At temperature 0, completions still often differ from the same server with speculation off: draft verification runs the target model through a different attention path than one-token decoding, and the resulting small numeric differences flip which of two near-tied candidate tokens wins. Near-ties are common in this model's reasoning text, so the greedy text often diverges early in the reasoning. KV-cache prefix reuse and batch composition change temperature-0 completions in the same way; at temperature > 0 the effect disappears into ordinary sampling variation.

Suffix-automaton (SA) drafting — model-free speculation that drafts from the request's own token history — is also available for low-concurrency or copy-heavy serving. Its configuration and measured trade-offs are covered in the [Muse Spark Open speculative-decoding guide](deployment-guide-for-musespark-open-speculative-decoding-on-trtllm.md).

## Testing API Endpoint

### Health Check

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When the `Status: 200` code is returned, the server is ready for queries. Note that the very first query may take longer due to initialization and compilation.

### Basic Test

After the TensorRT LLM server shows `Application startup complete`, you can send requests to the server.

The checkpoint's chat template has four reasoning efforts: `low`, `medium`, `high`, and `xhigh`, the default. Reasoning tokens count toward `max_tokens`: at the default effort, a small `max_tokens` can be spent entirely on reasoning and leave `content` empty. The examples below set `"reasoning_effort": "low"`; raise the effort and `max_tokens` together for harder prompts. The server maps the other OpenAI effort values onto the template's levels: `max` becomes `xhigh`, and `none` and `minimal` become `low`, because the template cannot turn reasoning off.

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/musespark-open-checkpoint",
    "messages": [
        {
            "role": "user",
            "content": "Where is New York?"
        }
    ],
    "max_tokens": 1024,
    "temperature": 0,
    "reasoning_effort": "low"
}' -w "\n"
```

The response should contain a `choices[0].message.content` field completing the request, plus a `usage` section with the token counts. With `--reasoning_parser musespark`, the model's reasoning is returned separately in `choices[0].message.reasoning_content`.

### Guided Decoding Test

With `guided_decoding_backend: xgrammar` in the configuration, constrained JSON output is available through the standard `response_format` field:

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "/path/to/musespark-open-checkpoint",
    "messages": [{"role": "user", "content": "Give me a city and its country as JSON."}],
    "max_tokens": 1024,
    "reasoning_effort": "low",
    "response_format": {"type": "json_object"}
}' -w "\n"
```

With `--reasoning_parser musespark`, the model writes its reasoning before the constrained answer, so the reasoning tokens count toward `max_tokens` here as well; the JSON object is returned in `choices[0].message.content`.

## Benchmarking Performance

Launch the server with the deployment YAML that you want to measure, as described in [Launch the TensorRT LLM Server](#launch-the-tensorrt-llm-server). From a machine that can reach the server, choose the input length, output length, and concurrency for your workload:

```bash
MODEL=/path/to/musespark-open-checkpoint
HOST=localhost
PORT=8000
ISL=8192
OSL=1024
CONCURRENCIES="1 32 256"
NUM_ROUNDS=5
RESULT_DIR=/path/to/musespark-open-results

mkdir -p "${RESULT_DIR}"

for concurrency in ${CONCURRENCIES}; do
    python3 -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model "${MODEL}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --dataset-name random \
        --random-ids \
        --tokenize-on-client \
        --random-input-len "${ISL}" \
        --random-output-len "${OSL}" \
        --num-prompts "$((concurrency * NUM_ROUNDS))" \
        --max-concurrency "${concurrency}" \
        --ignore-eos \
        --trust-remote-code \
        --save-result \
        --result-dir "${RESULT_DIR}" \
        --result-filename "concurrency_${concurrency}.json"
done
```

Before running the benchmark, adjust `ISL` and `OSL` for the server configuration being measured. Their sum must not exceed the server's `max_seq_len`.

### Key Metrics

#### Time to First Token (TTFT)
  * The typical time elapsed from when a request is sent until the first output token is generated.

#### Time Per Output Token (TPOT) and Inter-Token Latency (ITL)
  * TPOT is the typical time required to generate each token *after* the first one.
  * ITL is the typical time delay between the completion of one token and the completion of the next.
  * Both TPOT and ITL ignore TTFT.

#### End-to-End (E2E) Latency
  * The typical total time from when a request is submitted until the final token of the response is received.

#### Total Token Throughput
  * The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens.

## Troubleshooting Tips

* **Startup fails with a paged-context FMHA error naming head dimension 64 on `SM103`:** chunked prefill, KV-cache block reuse, and speculative decoding require the FP8 KV cache for this model on `SM103` (`kv_cache_config.dtype: fp8`); a 16-bit KV cache in that combination is refused by design. Either keep the FP8 KV cache (recommended) or disable chunked prefill, block reuse, and speculative decoding.
* **CUDA out-of-memory errors:** Reduce `kv_cache_config.free_gpu_memory_fraction`, `max_batch_size`, or `max_seq_len`.
* **Disaggregated serving produces attention-kernel errors:** Set `TLLM_FMHA_LIBS=-fallback` in the server environment for the affected configuration.
* **Startup refuses the cache transceiver configuration:** keep the defaults (`backend: NIXL`, `transceiver_runtime: PYTHON`); the model needs the KV cache manager v2, and disaggregated serving supports it only on that route.
* **Model fails to load:** Make sure the checkpoint path is a complete Hugging Face-format Muse Spark Open checkpoint with its tokenizer and processor files.
* **Tool calls or reasoning content not parsed:** Pass `--tool_parser musespark` and `--reasoning_parser musespark` on the `trtllm-serve` command line; `auto` resolves to `musespark` only for a local checkpoint directory.
* **GPU utilization:** For performance issues, check GPU utilization with `nvidia-smi` while the server is running.
