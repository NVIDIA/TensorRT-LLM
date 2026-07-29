# Deployment Guide for MiniMax-M3 on TensorRT LLM

## Introduction

This deployment guide provides step-by-step instructions for running the MiniMax-M3 model using TensorRT LLM. It covers the complete setup required; from accessing model weights and preparing the software environment to configuring TensorRT LLM parameters, launching the server, and validating inference output.

MiniMax-M3 is a Mixture-of-Experts (MoE) model that uses MiniMax block-sparse attention. The first few layers use dense attention with a dense MLP, while the remaining layers combine a sparse attention path (an index-K block selector followed by sparse grouped-query attention) with MoE (top-4 of 128 routed experts plus one shared expert). In TensorRT LLM it is served through the `MiniMaxM3SparseForConditionalGeneration` architecture (text, image, and video) and the text-only `MiniMaxM3SparseForCausalLM` architecture.

TensorRT LLM supports two precisions for MiniMax-M3:

* **BF16** — the official upstream checkpoint from MiniMaxAI.
* **MXFP8** — an NVIDIA-published checkpoint that quantizes the MoE/Linear weights to MXFP8 while keeping activations and the KV cache in BF16. The weights occupy ~half the memory of BF16, which is the recommended choice for throughput-oriented deployments and is the default for this guide.

The block-sparse attention path does **not** currently support KV cache reuse or Multi-Token Prediction (MTP) in this release.

This guide deploys MiniMax-M3 on **8x NVIDIA GB200 GPUs across 2 nodes** (4 GPUs per node) using Slurm and the `trtllm-llmapi-launch` multi-node launcher, with the MoE experts distributed via expert parallelism. The attention layers can run with either Tensor-Expert Parallelism (TEP) or Data-Expert Parallelism (DEP); see [Choosing the Parallelism Strategy](#choosing-the-parallelism-strategy-tep-vs-dep).

The guide is intended for developers and practitioners seeking high-throughput or low-latency inference using NVIDIA's accelerated stack.

## Prerequisites

* GPU: 8x NVIDIA GB200 GPUs across 2 nodes (4 GPUs per node). Tensor/expert parallelism of 8 (`tensor_parallel_size: 8`, `moe_expert_parallel_size: 8` in the curated YAML) spans all 8 GPUs. Plan for the corresponding memory footprint — MXFP8 weights occupy roughly half of BF16.
* Multi-node launcher: Slurm with the pyxis/enroot container plugin (or an equivalent MPI launcher) to start one rank per GPU across both nodes.
* High-speed inter-node interconnect (e.g., InfiniBand) for tensor/expert-parallel traffic.
* Shared filesystem visible to both nodes for the model weights and the configuration file.
* OS: Linux
* Drivers: CUDA Driver 575 or later
* Container runtime with NVIDIA GPU support on each node

## Models

Two checkpoints are supported. Both are loaded through the same `MiniMaxM3SparseForConditionalGeneration` / `MiniMaxM3SparseForCausalLM` architectures and share the same chat template and serving CLI; the only difference is the on-disk weight format.

### MXFP8 (recommended for throughput)

* [MiniMaxAI/MiniMax-M3-MXFP8](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8) — MiniMaxAI-published MXFP8-quantized checkpoint. Weights are stored in MXFP8 (block size 1×32); activations and the KV cache stay in BF16.

```bash
git lfs install
git clone https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8 /models/MiniMax-M3-MXFP8
```

### BF16 (upstream)

* [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) — Official BF16 checkpoint from MiniMaxAI.

```bash
git lfs install
git clone https://huggingface.co/MiniMaxAI/MiniMax-M3 /models/MiniMax-M3
```

Both checkpoints ship their own chat template (`chat_template.jinja`), which is passed explicitly to the server (see [Launch the TensorRT LLM Server](#launch-the-tensorrt-llm-server)).

## Feature Support Notes

* **Block-sparse attention is required.** MiniMax-M3 runs on the block-sparse attention backend, which must be selected via `sparse_attention_config.algorithm: minimax_m3` in the YAML configuration. There is no dense fallback for the sparse layers.
* **Supported precisions: BF16 and MXFP8.** No additional FP8/NVFP4 serving paths are supported at this time. MXFP8 quantizes only the weights; activations and the KV cache stay in BF16, so the curated YAML is identical for both checkpoints. The default MoE backend is used.
* **KV cache reuse must be disabled.** KV cache reuse is not supported on the sparse-attention path, so set `kv_cache_config.enable_block_reuse: false`.
* **MTP is not supported** on the sparse-attention path in this release.
* **`max_seq_len` must be capped for CUDA graphs.** The dense GQA expansion in the first few attention layers and the per-Q FP32 expansion in the sparse decode kernel allocate temporary tensors whose size grows linearly with the warmup decode's `max_k`. If `max_seq_len` is left at the checkpoint default, that `max_k` follows `max_position_embeddings` (1M for MXFP8, 512K for BF16) and the resulting gigabyte-scale single-allocation request exceeds the caching allocator's CUDA-graph-safe path, so capture fails with `cudaErrorStreamCaptureUnsupported` / OOM. The curated YAML therefore sets `max_seq_len` to a small value just above ISL+OSL (`2068` for the 1k/1k benchmark) so CUDA graphs can capture cleanly. Raise it for longer-context workloads, but expect a corresponding cut in `max_batch_size`.
* **Parallelism.** MoE experts run with expert parallelism. The attention layers support both Tensor-Expert Parallelism (TEP) and Data-Expert Parallelism (DEP, via `enable_attention_dp: true`). The overlap scheduler is enabled by default.
* **Multimodal.** `MiniMaxM3SparseForConditionalGeneration` supports text, image, and video inputs. The text decoder is also usable standalone (text-only) via the `MiniMaxM3SparseForCausalLM` architecture.

## Deployment Steps

MiniMax-M3 is deployed across 2 nodes (8x GB200 total) using Slurm with the pyxis/enroot container plugin. The model weights and the configuration file must live on a **shared filesystem** visible to both nodes.

### Container Image

The TensorRT LLM NVIDIA NGC image is used as the Slurm container:

```text
nvcr.io/nvidia/tensorrt-llm/release:x.y.z
```

Note:

* See <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags> for all available containers. Containers published in the main branch weekly have an `rcN` suffix, while the monthly release with QA tests has no `rcN` suffix. Use the `rc` release to get the latest model and feature support.
* If you want to use the latest main branch, you can build from source: [https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html](https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html)

### Recommended Performance Settings

**Treat these as a starting point and tune the parameters for your workload.**

We provide a curated YAML configuration in the TensorRT LLM repository that bundles the MiniMax-M3 sparse-attention backend, the KV-cache settings required by the sparse path, and a tuned set of throughput knobs:

```shell
TRTLLM_DIR=/app/tensorrt_llm # change as needed to match your environment
EXTRA_LLM_API_FILE=${TRTLLM_DIR}/examples/configs/curated/minimax-m3-throughput.yaml
```

If you don't have access to the source code locally, you can manually create the YAML config file using the code in the dropdown below.

````{admonition} Show MiniMax-M3 throughput config
:class: dropdown

```{literalinclude} ../../../examples/configs/curated/minimax-m3-throughput.yaml
:language: yaml
```
````

The configuration uses Data-Expert Parallelism (DEP): `enable_attention_dp: true` runs the attention layers data-parallel across ranks while the MoE experts run expert-parallel, which favors high-throughput / large-batch serving on MiniMax-M3.

### Launch the TensorRT LLM Server

MiniMax-M3 is launched through the `trtllm-llmapi-launch` wrapper, which sets up the multi-rank (MPI/Slurm) environment that the parallel server requires. The wrapper is run once per rank by Slurm (`srun`), with one task (rank) per GPU. The example below launches the server across 2 nodes (`-N 2`), 4 GPUs per node (`--ntasks-per-node 4`, 8 ranks total), using the curated YAML to drive parallelism, batching, and the MiniMax-M3 sparse-attention backend:

```bash
export MODEL=/models/MiniMax-M3-MXFP8   # MXFP8 (recommended); use /models/MiniMax-M3 for BF16

srun -N 2 \
    --ntasks 8 --ntasks-per-node 4 \
    --mpi=pmix --gres=gpu:4 \
    --container-image=nvcr.io/nvidia/tensorrt-llm/release:x.y.z \
    --container-mounts=/models:/models,/workspace:/workspace,${TRTLLM_DIR}:${TRTLLM_DIR} \
    --container-workdir /workspace \
    bash -c "trtllm-llmapi-launch \
        python3 -m tensorrt_llm.commands.serve $MODEL \
          --trust_remote_code \
          --reasoning_parser minimax_m3 \
          --tool_parser minimax_m3 \
          --chat_template $MODEL/chat_template.jinja \
          --host 0.0.0.0 \
          --extra_llm_api_options $EXTRA_LLM_API_FILE"
```

The parallelism, batch, KV-cache, sparse-attention, and CUDA-graph settings all live in the YAML; no CLI flags need to change to tune them. The same `$EXTRA_LLM_API_FILE` is used for both the MXFP8 and BF16 checkpoints — point `$MODEL` at the directory of whichever checkpoint you want to serve.

> [!NOTE]
> Adjust `-N`, `--ntasks`, `--ntasks-per-node`, and `--gres=gpu:` to match your cluster's GPUs-per-node. The total number of tasks (ranks) must equal `tensor_parallel_size` (`8`). Add the partition / account / node-list flags (`-p`, `-A`, `-w`) required by your Slurm setup, and ensure `/models`, `/workspace`, and the TensorRT LLM repository resolve to the same shared paths on both nodes.

### Command-Line and YAML Options

#### Command-line options

* `--trust_remote_code`: Required to load the MiniMax-M3 configuration and custom code from the checkpoint.
* `--reasoning_parser minimax_m3`: Parses the MiniMax-M3 reasoning trace, which is wrapped in `<mm:think>...</mm:think>` tags, into the structured `reasoning_content` field of chat responses.
* `--tool_parser minimax_m3`: Parses MiniMax-M3 tool/function calls from the model output.
* `--chat_template`: Path to the chat template shipped with the checkpoint (`chat_template.jinja`).
* `--extra_llm_api_options`: Path to the YAML configuration file with the LLM API options described below.

#### `--extra_llm_api_options` YAML options

The curated config sets all the YAML knobs; this section only documents the two MiniMax-M3-specific constraints. Every other field in the YAML is a standard TensorRT LLM throughput knob — see the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full reference.

##### `sparse_attention_config.algorithm`

Must be set to `minimax_m3`. There is no dense fallback for the MiniMax-M3 sparse-attention layers.

##### `kv_cache_config.enable_block_reuse`

Must be `false`. The sparse-attention path does not support KV cache block reuse across requests with shared prefixes.

## Testing API Endpoint

The server (the OpenAI-compatible REST endpoint) runs on the rank-0 node, listening on port `8000`. Send requests to that node's hostname or IP; `localhost` only works from the rank-0 node itself. The examples below use `localhost:8000` — replace it with the rank-0 node address when querying from elsewhere.

### Health Check

You can query the health/readiness of the server using:

```bash
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When `Status: 200` is returned, the server is ready for queries. Note that the very first query may take longer due to initialization and compilation.

### Basic Test

After the TensorRT LLM server is set up and shows *Application startup complete*, you can send requests to the server. MiniMax-M3 is a reasoning model, so use the `/v1/chat/completions` endpoint — that path applies the chat template and, together with the configured reasoning parser, returns the reasoning trace separately in `reasoning_content`. (The `/v1/completions` endpoint does not apply the chat template and is not recommended for this model.)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "MiniMaxAI/MiniMax-M3-MXFP8",
      "messages": [
          {"role": "user", "content": "What is the capital of France?"}
      ],
      "max_tokens": 256,
      "temperature": 0
  }'
```

Example response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "MiniMaxAI/MiniMax-M3-MXFP8",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is **Paris**.",
        "reasoning_content": "The user is asking a simple factual question..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 134,
    "total_tokens": 181,
    "completion_tokens": 47
  }
}
```

The `message` object contains the visible answer in `content` and, when the model emits a `<mm:think>...</mm:think>` block, the parsed `reasoning_content`.

## Troubleshooting Tips

* **Sparse attention not applied:** Ensure `sparse_attention_config.algorithm` is set to `minimax_m3` in the YAML file passed to `--extra_llm_api_options`.
* **KV cache reuse errors:** Confirm `kv_cache_config.enable_block_reuse` is `false`; KV cache reuse is not supported on the sparse-attention path.
* **Model fails to load:** Make sure `--trust_remote_code` is set and that the checkpoint path is correct and fully downloaded (Git LFS).
* **Multi-node startup hangs or ranks can't find each other:** Verify that the model weights, the curated YAML, and all mounted paths resolve identically on both nodes (shared filesystem), that the total Slurm task count equals `tensor_parallel_size` (`8`), and that the inter-node interconnect (e.g., InfiniBand) is healthy.
* **Reasoning/tool output not parsed:** Verify `--reasoning_parser minimax_m3`, `--tool_parser minimax_m3`, and `--chat_template $MODEL/chat_template.jinja` are all passed.
* **GPU utilization:** For performance issues, check GPU utilization with `nvidia-smi` while the server is running.
* **Container startup:** If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed.
* **Port conflicts:** Make sure the server port (`8000` in this guide) is not being used by another application.
* **Configuration files:** Ensure that YAML config files are correctly formatted to avoid runtime errors.

## Benchmarking Performance

To benchmark the performance of your TensorRT LLM server, you can use the built-in `benchmark_serving.py` script. First, create a wrapper `bench.sh` script:

```bash
cat << 'EOF' > bench.sh
concurrency_list="32 64 128 256 512 1024"
multi_round=5
isl=1024
osl=1024
result_dir=/tmp/minimax_m3_output

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model MiniMaxAI/MiniMax-M3-MXFP8 \
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

To save results to files, add these options to each benchmark command:

```bash
--save-result \
--result-dir "${result_dir}" \
--result-filename "concurrency_${concurrency}.json"
```

For more benchmarking options see [benchmark_serving.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py).

Run `bench.sh` to begin a serving benchmark. This will take a long time if you run all the concurrencies.

```bash
./bench.sh
```

Sample TensorRT LLM serving benchmark output. Your results may vary due to ongoing software optimizations.

```
============ Serving Benchmark Result ============
Successful requests:                      [result]
Benchmark duration (s):                   [result]
Total input tokens:                       [result]
Total generated tokens:                   [result]
Request throughput (req/s):               [result]
Output token throughput (tok/s):          [result]
Total Token throughput (tok/s):           [result]
User throughput (tok/s):                  [result]
---------------Time to First Token----------------
Mean TTFT (ms):                           [result]
Median TTFT (ms):                         [result]
P99 TTFT (ms):                            [result]
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                           [result]
Median TPOT (ms):                         [result]
P99 TPOT (ms):                            [result]
---------------Inter-token Latency----------------
Mean ITL (ms):                            [result]
Median ITL (ms):                          [result]
P99 ITL (ms):                             [result]
----------------End-to-end Latency----------------
Mean E2EL (ms):                           [result]
Median E2EL (ms):                         [result]
P99 E2EL (ms):                            [result]
==================================================
```

### Key Metrics

#### Time to First Token (TTFT)
The typical time elapsed from when a request is sent until the first output token is generated.

#### Time Per Output Token (TPOT) and Inter-Token Latency (ITL)
* TPOT is the typical time required to generate each token *after* the first one.
* ITL is the typical time delay between the completion of one token and the completion of the next.
* Both TPOT and ITL ignore TTFT.

For a single request, ITLs are the time intervals between tokens, while TPOT is the average of those intervals:

$$
\text{TPOT (1 request)} = \text{Avg(ITL)} = \frac{\text{E2E latency} - \text{TTFT}}{\text{Num Output Tokens} - 1}
$$

Across different requests, **average TPOT** is the mean of each request's TPOT (all requests weighted equally), while **average ITL** is token-weighted (all tokens weighted equally):

$$
\text{Avg TPOT (N requests)} = \frac{\text{TPOT}_1 + \text{TPOT}_2 + \cdots + \text{TPOT}_N}{N}
$$

$$
\text{Avg ITL (N requests)} = \frac{\text{Sum of all ITLs across requests}}{\text{Num Output Tokens across requests}}
$$

#### End-to-End (E2E) Latency
The typical total time from when a request is submitted until the final token of the response is received.

#### Total Token Throughput
The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens.

$$
\text{Total TPS} = \frac{\text{Num Input Tokens}+\text{Num Output Tokens}}{T_{last} - T_{first}}
$$

#### Tokens Per Second (TPS) or Output Token Throughput
How many output tokens the system generates each second.

$$
\text{TPS} = \frac{\text{Num Output Tokens}}{T_{last} - T_{first}}
$$
