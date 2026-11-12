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

* A complete Hugging Face-format Muse Spark Open checkpoint and tokenizer, e.g. `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` (`model_type: muse_spark`, architecture `MuseSparkForConditionalGeneration`). The checkpoint ships the routed-expert weights pre-quantized to NVFP4; no additional quantization step is needed.

The examples below use `/path/to/musespark-open-checkpoint` — replace it with your local path or the Hub id. The Hub repository is gated: request access on its Hugging Face page and set `HF_TOKEN` before serving it by id.

## Feature Support Notes

* **Blackwell (`SM100` family) GPUs are supported.** The results and constraints in this guide were validated on GB300 NVL (`SM103`).
* **Aggregated serving with TP4/EP4 is the validated deployment.** Attention runs tensor-parallel (`enable_attention_dp: false`) and the 256 experts shard 4-way (`moe_expert_parallel_size: 4`).
* **FP8 KV cache is required with chunked prefill or KV reuse on `SM103`.** For head dimension 64 on `SM103`, the fused paged-context FMHA kernel exists only for the FP8 KV-cache dtype. Paged-context FMHA is exercised whenever a request enters attention with a non-zero cached prefix — chunked prefill, KV-cache block reuse, or speculative draft verification — so those features require `kv_cache_config.dtype: fp8` on `SM103`; a 16-bit KV cache in that combination is refused at startup rather than risking silent accuracy loss. The recommended configuration below satisfies this.
* **CUDA graphs and the overlap scheduler are enabled** (`disable_overlap_scheduler: false`, `cuda_graph_config.max_batch_size: 256` with `enable_padding: true`).
* **Chunked prefill is supported and enabled** (`enable_chunked_prefill: true`).
* **KV-cache block reuse is supported and enabled** (`enable_block_reuse: true`, `enable_partial_reuse: true`).
* **Guided decoding is supported** with the `xgrammar` backend (`guided_decoding_backend: xgrammar`), including JSON-schema constrained output.
* **Speculative decoding is supported** with a DFlash2 drafter, and model-free suffix-automaton (SA) drafting is available as an opt-in; see [Speculative Decoding](#speculative-decoding) below.
* **Agentic workloads** (long multi-turn sessions, up to the full 1M-token context) have two dedicated configurations; see [Agentic Workloads](#agentic-workloads) below.
* **Disaggregated serving:** this model runs on the KV cache manager v2 only, which disaggregated serving keeps on the NIXL backend with the Python transceiver runtime (`cache_transceiver_config: {backend: NIXL, transceiver_runtime: PYTHON}`); other transceiver configurations are refused at startup. Some disaggregated configurations additionally require adjusting the attention-kernel library selection with `TLLM_FMHA_LIBS=-fallback` in the server environment (this removes the fallback attention library from the default selection order). On GB300 (`SM103`, aarch64) also set `UCX_TLS=cuda_copy,cuda_ipc,sm,self,tcp` in every worker environment: with the default UCX transport list, NIXL memory registration fails at startup (`registerMem: registration failed for the specified or all potential backends`).

## Deployment Steps

### Run Docker Container

Run the Docker container using the TensorRT LLM NVIDIA NGC image published from this branch. The image is multi-arch (x86_64 and aarch64), so it runs on GB300 trays directly:

```bash
docker run --rm -it \
    --ipc=host \
    --gpus all \
    -p 8000:8000 \
    -v "$HOST_MODELS_DIR":/models \
    --name tensorrt_llm \
    nvcr.io/nvidia/tensorrt-llm/release:1.4.0a0.dev202610171236 \
    /bin/bash
```

> [!WARNING]
> `1.4.0a0.dev202610171236` is a preview tag published from the head of `feat/musespark-open` (commit `79e45c5d`) on 2026-10-17. Preview artifacts are retained for roughly two months, so expect this tag (and the matching wheels) to be removed around mid-December 2026. If the tag is gone, use a newer release published from this branch (see the [release tags page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags)) or build the container from this branch's sources (see the [Docker guide](https://github.com/NVIDIA/TensorRT-LLM/blob/feat/musespark-open/docker/README.md)).

For installations without Docker, the matching Python wheel is available from the preview index:

```bash
pip install "tensorrt_llm==1.4.0a0.dev202610171236" --extra-index-url https://pypi.nvidia.com/trtllm_preview/
```

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
    --custom_tokenizer musespark_open \
    --tool_parser musespark \
    --reasoning_parser musespark
```

The `--tool_parser` / `--reasoning_parser` flags enable tool-call and reasoning-content extraction on `/v1/chat/completions`; pass `musespark` explicitly for this model.

`--custom_tokenizer musespark_open` selects the model's bundled tokenizer wrapper (a registered alias); it is not auto-detected, and the speculative-decoding guide lists it as a prerequisite. The wrapper carries the model's padding and dual EOS-id configuration, which a generic Hugging Face tokenizer does not.

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

* **Description:** Required to load the tokenizer code shipped with the checkpoint.

## Multimodal Inputs

Muse Spark Open accepts image, video, and audio inputs alongside text:

* **Image and video** are encoded by the checkpoint's vision tower (14x14 patches, 2x2 spatial merge, 2-frame temporal patching). Per the checkpoint's processor configuration, an image contributes up to 4,096 tokens and a video contributes up to 256 tokens per sampled frame (2 frames per second, up to 600 frames).
* **Audio** is encoded by the convolutional audio tower from a 24 kHz feature extractor; each audio token covers 1,920 input samples (80 ms).

Token budgets for multimodal items count against `max_num_tokens` and `max_seq_len` like text tokens; size long-video requests accordingly.

## Speculative Decoding

Muse Spark Open supports speculative decoding with a **DFlash2 drafter** — a small draft model that cross-attends to hidden states captured from the target model and predicts multiple draft tokens in one forward pass. The published drafter checkpoint is [`nvidia/Muse-Spark-Open-816B-A42B-DFlash2`](https://huggingface.co/nvidia/Muse-Spark-Open-816B-A42B-DFlash2). A typical configuration drafts `K = 7` tokens per step:

```yaml
speculative_config:
  decoding_type: DFlash
  max_draft_len: 7
  speculative_model: nvidia/Muse-Spark-Open-816B-A42B-DFlash2
```

DFlash runs on the `TRTLLM` attention backend — the backend the validated aggregated recipe above already ships; a FlashInfer-attention configuration is refused at startup.

Draft-token acceptance is strict: an accepted token is exactly the token the target model emits at that position. At temperature 0, completions still often differ from the same server with speculation off: draft verification runs the target model through a different attention path than one-token decoding, and the resulting small numeric differences flip which of two near-tied candidate tokens wins. Near-ties are common in this model's reasoning text, so the greedy text often diverges early in the reasoning. KV-cache prefix reuse and batch composition change temperature-0 completions in the same way; at temperature > 0 the effect disappears into ordinary sampling variation.

Suffix-automaton (SA) drafting — model-free speculation that drafts from the request's own token history — is also available as an opt-in for low-concurrency or copy-heavy serving. Its measured trade-offs are covered in the [Muse Spark Open speculative-decoding guide](deployment-guide-for-musespark-open-speculative-decoding-on-trtllm.md).

## Agentic Workloads

Agentic coding sessions resend their whole conversation with every request. In the AgentX runs measured below, the median request carried 90,000 to 100,000 input tokens and the 99th percentile 500,000 to 670,000, nearly all of it the context of an earlier request in the same session, and the median response was about 330 tokens. Serving this traffic is mostly a matter of keeping and reusing the KV cache of earlier turns. `examples/musespark_open/` has two configurations for it, measured on one GB300 NVL node:

| Configuration | Concurrent sessions | Total throughput (tokens/s/GPU) | Output speed per session (tokens/s, p50 / p90) | TTFT (s, p50 / p90) |
|---|---|---|---|---|
| [`serve_config_agentx_max_throughput.yaml`](https://github.com/NVIDIA/TensorRT-LLM/blob/feat/musespark-open/examples/musespark_open/serve_config_agentx_max_throughput.yaml) | 48 | 26,169 | 24.1 / 64.6 | 1.3 / 2.7 |
| [`serve_config_agentx_balanced.yaml`](https://github.com/NVIDIA/TensorRT-LLM/blob/feat/musespark-open/examples/musespark_open/serve_config_agentx_balanced.yaml) | 24 | 19,306 | 41.9 / 108.6 | 0.9 / 2.2 |

Use the max-throughput configuration to serve the most sessions per node, and the balanced one when each session should get its first token and its output sooner. Launch either one in place of the recommended settings above:

```bash
MODEL=/path/to/musespark-open-checkpoint
CONFIG=/path/to/serve_config_agentx_max_throughput.yaml   # from examples/musespark_open/

trtllm-serve ${MODEL} \
    --host 0.0.0.0 --port 8000 \
    --config ${CONFIG} \
    --custom_tokenizer musespark_open \
    --tool_parser musespark \
    --reasoning_parser musespark
```

Startup takes about 6 to 10 minutes, most of it spent capturing CUDA graphs for the decode batch sizes and the prefill token counts.

### What the Agentic Configurations Change

Compared with the recommended settings above:

* **Full context.** `max_seq_len: 1048576`, the model's context window.
* **KV-cache reuse tuned for long sessions** (KV cache manager v2). Blocks become reusable after every prefill chunk rather than at the end of the prompt (`block_reuse_config.policy: all_reusable`); under cache pressure, the sliding-window blocks that end each prompt are kept longest (`swa_endpoint_rewind_tokens: 1024`); and the GPU KV pool is split at a fixed ratio between the sliding-window and the full-attention layers (`pool_ratio`).
* **A host-memory KV tier.** `host_cache_size` gives each GPU 160 GiB of host memory for KV blocks evicted from GPU memory, so a session that comes back copies its context back instead of recomputing it. The node needs 640 GiB of host memory for this tier in addition to the server processes: in the measured runs, host memory use peaked at about 820 GiB of 950 GiB. Lower `host_cache_size` on hosts with less.
* **Prefill CUDA graphs** (`prefill_cuda_graph_backend: breakable` with `prefill_capture_num_tokens`) and the `MAX_UTILIZATION` capacity-scheduler policy.
* **Serving-side settings:** `num_postprocess_workers: 8`, `stream_interval: 20` and `enable_tokenization_cache: true`. Postprocess workers apply to `trtllm-serve`; with the LLM API (for example `trtllm-eval`), leave `num_postprocess_workers` unset, or the outputs come back without text.
* **Text only.** `disable_mm_encoder: true` skips loading the vision and audio towers, which leaves more GPU memory for the KV cache; image, video and audio requests are rejected. Remove the line to serve multimodal requests.
* **No speculative decoding.** On this branch, the DFlash2 drafter keeps a dense context buffer for every batch slot, sized for `max_seq_len`: about 5 GiB per GPU per slot at 1M tokens, which does not fit next to these batch sizes.

Raising concurrency past 48 sessions buys little. In 20-minute runs with these settings (the hour-long results below are the reference), 56 and 64 sessions added 2 % and 5 % throughput, while the p90 time to first token rose from 2.9 s to 8.8 s and 11.0 s:

| Concurrent sessions (= `max_batch_size`) | 16 | 24 | 32 | 40 | 48 | 56 | 64 |
|---|---|---|---|---|---|---|---|
| Total throughput (tokens/s/GPU) | 13,415 | 16,188 | 22,108 | 24,486 | 27,639 | 28,185 | 29,084 |
| Output speed per session (tokens/s, p50) | 53.4 | 34.3 | 30.2 | 27.4 | 25.1 | 21.4 | 18.4 |
| TTFT p90 (s) | 1.6 | 2.9 | 6.6 | 2.6 | 2.9 | 8.8 | 11.0 |

With the same engine settings, minus the serving-side ones that `trtllm-eval` does not take, GSM8K scores 94.47 with the max-throughput configuration and 95.22 with the balanced one.

### Measuring with AgentX

The numbers come from the AgentX scenario of SemiAnalysis's [agentx-harness](https://github.com/SemiAnalysisAI/agentx-harness) (aiperf 0.12.0), which replays the public dataset [`semianalysisai/cc-traces-weka-062126`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) (Apache-2.0): 393 Claude Code sessions with their sub-agents, recorded as each request's input and output length and its prefix overlap with earlier requests. To measure a server the same way:

```bash
pip install "git+https://github.com/SemiAnalysisAI/agentx-harness.git@104c7c8086666385fbf3962911b55f68bf08a04a"

aiperf profile --scenario agentx \
    --url http://localhost:8000 --endpoint /v1/chat/completions --endpoint-type chat --streaming \
    --model ${MODEL} --tokenizer ${MODEL} --tokenizer-trust-remote-code \
    --public-dataset semianalysis_cc_traces_weka_062126 --num-dataset-entries 393 \
    --max-context-length 1048576 --concurrency 48 \
    --benchmark-duration 3600 --benchmark-grace-period 30 \
    --warmup-requests-per-lane 10 --warmup-grace-period 1800 \
    --trace-idle-gap-cap-seconds 300 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --use-server-token-count --random-seed 42 --failed-request-threshold 0.10 \
    --stats-interval 30 --slice-duration 1.0 --no-gpu-telemetry \
    --output-artifact-dir agentx-results
```

Notes on the measurement:

* Each replayed session starts between 25 % and 75 % of the way into its trace, and each of the concurrent sessions first replays 10 warm-up requests, so the measured hour starts with a populated cache.
* Total throughput counts input and output tokens; about 96 % of the input tokens are served from the prefix cache.
* The chat template writes the current date into each prompt, so a date change during a run invalidates every cached prefix. Start the server with a `TZ` whose date does not change during the run.
* With `--reasoning_parser musespark`, a response whose few output tokens are all reasoning has no `content`, and aiperf counts it as a failed request.

Results of one measured hour per configuration (2026-11-11), on one GB300 NVL node (4 GPUs) at commit `0e868637c8` of this branch, with the checkpoint `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` (revision `4239e4db`):

| Configuration | Sessions | Total (tokens/s/GPU) | Output (tokens/s/GPU) | Output speed per session (tokens/s, p50 / p90) | TTFT (s, p50 / p90 / p99) | Inter-token latency p50 (ms) | Prefix-cache hits | Failed requests |
|---|---|---|---|---|---|---|---|---|
| Max-throughput | 48 | 26,169 | 175 | 24.1 / 64.6 | 1.3 / 2.7 / 9.7 | 41.3 | 96.0 % | 6 of 2,969 (0.2 %) |
| Balanced | 24 | 19,306 | 143 | 41.9 / 108.6 | 0.9 / 2.2 / 6.7 | 23.4 | 96.6 % | 5 of 2,239 (0.2 %) |

## Testing API Endpoint

### Health Check

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When the `Status: 200` code is returned, the server is ready for queries. Note that the very first query may take longer due to initialization and compilation.

### Basic Test

After the TensorRT LLM server shows `Application startup complete`, you can send requests to the server.

The checkpoint's chat template defaults to the highest reasoning effort (`xhigh`), and reasoning tokens count toward `max_tokens`: at the default effort, a small `max_tokens` can be spent entirely on reasoning and leave `content` empty. The examples below set `"reasoning_effort": "low"`; raise the effort (`medium`, `high`, `xhigh`) and `max_tokens` together for harder prompts.

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

* **Startup fails with a paged-context FMHA error:** On `SM103`, chunked prefill and KV-cache block reuse require the FP8 KV cache for this model (`kv_cache_config.dtype: fp8`); a 16-bit KV cache in that combination is refused at startup by design. Either keep the FP8 KV cache (recommended) or disable both chunked prefill and block reuse.
* **CUDA out-of-memory errors:** Reduce `kv_cache_config.free_gpu_memory_fraction`, `max_batch_size`, or `max_seq_len`.
* **Disaggregated serving produces attention-kernel errors:** Set `TLLM_FMHA_LIBS=-fallback` in the server environment for the affected configuration.
* **Startup refuses the cache transceiver configuration (`requires KV cache manager v2`):** Use `cache_transceiver_config: {backend: NIXL, transceiver_runtime: PYTHON}`; the model does not run on the v1 KV cache manager, and that is the only disaggregated route that keeps v2.
* **Model fails to load:** Make sure `trust_remote_code: true` is set and that the checkpoint path is a complete Hugging Face-format Muse Spark Open checkpoint with its tokenizer and processor files.
* **Tool calls or reasoning content not parsed:** Pass `--tool_parser musespark` and `--reasoning_parser musespark` explicitly on the `trtllm-serve` command line.
* **GPU utilization:** For performance issues, check GPU utilization with `nvidia-smi` while the server is running.
