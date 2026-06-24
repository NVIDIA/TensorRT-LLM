# Deployment Guide for MiniMax-M3 on TensorRT LLM

## Introduction

This deployment guide provides step-by-step instructions for running the MiniMax-M3 model using TensorRT LLM. It covers the complete setup required; from accessing model weights and preparing the software environment to configuring TensorRT LLM parameters, launching the server, and validating inference output.

MiniMax-M3 is a Mixture-of-Experts (MoE) model that uses MiniMax block-sparse attention. The first few layers use dense attention with a dense MLP, while the remaining layers combine a sparse attention path (an index-K block selector followed by sparse grouped-query attention) with MoE (top-4 of 128 routed experts plus one shared expert). In TensorRT LLM it is served through the `MiniMaxM3SparseForConditionalGeneration` architecture (text, image, and video) and the text-only `MiniMaxM3SparseForCausalLM` architecture.

MiniMax-M3 is served in **BF16**; no FP8/NVFP4 serving path is supported at this time. The block-sparse attention path does **not** currently support KV cache reuse or Multi-Token Prediction (MTP) in this release.

This guide deploys MiniMax-M3 on **8x NVIDIA GB200 GPUs across 2 nodes** (4 GPUs per node) using Slurm and the `trtllm-llmapi-launch` multi-node launcher, with the MoE experts distributed via expert parallelism. The attention layers can run with either Tensor-Expert Parallelism (TEP) or Data-Expert Parallelism (DEP); see [Choosing the Parallelism Strategy](#choosing-the-parallelism-strategy-tep-vs-dep).

The guide is intended for developers and practitioners seeking high-throughput or low-latency inference using NVIDIA's accelerated stack.

## Prerequisites

* GPU: 8x NVIDIA GB200 GPUs across 2 nodes (4 GPUs per node). Tensor/expert parallelism of 8 (`--tp_size 8 --moe_expert_parallel_size 8`) spans all 8 GPUs, and the model is served in BF16, so plan for the corresponding memory footprint.
* Multi-node launcher: Slurm with the pyxis/enroot container plugin (or an equivalent MPI launcher) to start one rank per GPU across both nodes.
* High-speed inter-node interconnect (e.g., InfiniBand) for tensor/expert-parallel traffic.
* Shared filesystem visible to both nodes for the model weights and the configuration file.
* OS: Linux
* Drivers: CUDA Driver 575 or later
* Container runtime with NVIDIA GPU support on each node
* Minimum TensorRT LLM version: 1.3.0rc20

## Models

The following checkpoint is available:

* [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) — Official BF16 checkpoint

```bash
git lfs install
git clone https://huggingface.co/MiniMaxAI/MiniMax-M3 /models/MiniMax-M3
```

The checkpoint ships its own chat template (`chat_template.jinja`), which is passed explicitly to the server (see [Launch the TensorRT LLM Server](#launch-the-tensorrt-llm-server)).

## Feature Support Notes

* **Block-sparse attention is required.** MiniMax-M3 runs on the block-sparse attention backend, which must be selected via `sparse_attention_config.algorithm: minimax_m3` in the YAML configuration. There is no dense fallback for the sparse layers.
* **BF16 only.** MiniMax-M3 is served in BF16. No FP8/NVFP4 serving path is supported at this time. The default MoE backend is used.
* **KV cache reuse must be disabled.** KV cache reuse is not supported on the sparse-attention path, so set `kv_cache_config.enable_block_reuse: false`.
* **MTP is not supported** on the sparse-attention path in this release.
* **Parallelism**: MoE experts run with expert parallelism. The attention layers support both Tensor-Expert Parallelism (TEP) and Data-Expert Parallelism (DEP, via `--enable_attention_dp`). The overlap scheduler and CUDA graphs are also supported.
* **Multimodal.** `MiniMaxM3SparseForConditionalGeneration` supports text, image, and video inputs. The text decoder is also usable standalone (text-only) via the `MiniMaxM3SparseForCausalLM` architecture.

## Deployment Steps

MiniMax-M3 is deployed across 2 nodes (8x GB200 total) using Slurm with the pyxis/enroot container plugin. The model weights and the configuration file must live on a **shared filesystem** visible to both nodes.

### Container Image

The TensorRT LLM NVIDIA NGC image is used as the Slurm container:

```text
nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20
```

Note:

* See <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags> for all available containers. Containers published in the main branch weekly have an `rcN` suffix, while the monthly release with QA tests has no `rcN` suffix. Use the `rc` release to get the latest model and feature support.
* If you want to use the latest main branch, you can build from source: [https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html](https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source.html)

### Recommended Performance Settings

**Treat these as a starting point and tune the parameters for your workload.**

On the shared filesystem, create the extra LLM API options file that selects the MiniMax-M3 block-sparse attention backend and disables KV cache reuse:

```bash
cat > /workspace/extra_llm_api.yaml <<EOF
sparse_attention_config:
    algorithm: minimax_m3
kv_cache_config:
    enable_block_reuse: false
EOF
```

### Choosing the Parallelism Strategy (TEP vs DEP)

MiniMax-M3 distributes its MoE experts with expert parallelism (`--moe_expert_parallel_size 8`). You can pair this with either of two attention-parallelism strategies across the 8 GPUs:

* **DEP (Data-Expert Parallelism)** — add `--enable_attention_dp`. The attention layers run data-parallel (each rank processes a slice of the batch), while the MoE experts run expert-parallel. This is the configuration shown below, and typically favors high-throughput / large-batch serving.
* **TEP (Tensor-Expert Parallelism)** — omit `--enable_attention_dp`. The attention layers run tensor-parallel across all 8 ranks, while the MoE experts run expert-parallel. This can help latency-sensitive / small-batch workloads.

Both strategies use `--tp_size 8 --moe_expert_parallel_size 8`. Benchmark both for your workload before settling on one.

### Launch the TensorRT LLM Server

MiniMax-M3 is launched through the `trtllm-llmapi-launch` wrapper, which sets up the multi-rank (MPI/Slurm) environment that the parallel server requires. The wrapper is run once per rank by Slurm (`srun`), with one task (rank) per GPU. The example below launches the **DEP** configuration across 2 nodes (`-N 2`), 4 GPUs per node (`--ntasks-per-node 4`, 8 ranks total):

```bash
export MODEL=/models/MiniMax-M3   # path on the shared filesystem; mounted into the container

srun -N 2 \
    --ntasks 8 --ntasks-per-node 4 \
    --mpi=pmix --gres=gpu:4 \
    --container-image=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20 \
    --container-mounts=/models:/models,/workspace:/workspace \
    --container-workdir /workspace \
    bash -c "trtllm-llmapi-launch \
        python3 -m tensorrt_llm.commands.serve $MODEL \
          --tp_size 8 \
          --moe_expert_parallel_size 8 \
          --max_batch_size 128 \
          --enable_attention_dp \
          --free_gpu_memory_fraction 0.5 \
          --trust_remote_code \
          --reasoning_parser minimax_m3 \
          --tool_parser minimax_m3 \
          --chat_template $MODEL/chat_template.jinja \
          --host 0.0.0.0 \
          --extra_llm_api_options /workspace/extra_llm_api.yaml"
```

> [!NOTE]
> Adjust `-N`, `--ntasks`, `--ntasks-per-node`, and `--gres=gpu:` to match your cluster's GPUs-per-node. The total number of tasks (ranks) must equal `--tp_size` (`8`). Add the partition / account / node-list flags (`-p`, `-A`, `-w`) required by your Slurm setup, and ensure `/models` and `/workspace` resolve to the same shared paths on both nodes.
> To run the **TEP** configuration instead, remove the `--enable_attention_dp` flag.

> [!WARNING]
> If you encounter OOM errors, try one or more of the following:
> - Lower `--free_gpu_memory_fraction` (e.g., `0.4`).
> - Reduce `--max_batch_size` (e.g., `64` or `32`).
> - As a workaround for memory fragmentation, set `PYTORCH_ALLOC_CONF=max_split_size_mb:8192`. For more details, refer to the [PyTorch documentation on optimizing memory usage](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf).

### Command-Line and YAML Options

#### Command-line options

* `--tp_size`: Tensor parallel size. MiniMax-M3 is served with `8` (across the 2 nodes).
* `--moe_expert_parallel_size` (`--ep_size`): Expert parallel size for the MoE layers. Set to `8` to match the tensor-parallel size.
* `--max_batch_size`: Maximum number of requests batched together.
* `--enable_attention_dp`: Enables Data-Expert Parallelism (DEP) by running the attention layers data-parallel across ranks. Omit it to use Tensor-Expert Parallelism (TEP). See [Choosing the Parallelism Strategy](#choosing-the-parallelism-strategy-tep-vs-dep).
* `--free_gpu_memory_fraction`: Fraction of free GPU memory reserved for the KV cache after the model is loaded. **Recommendation:** If you experience OOM errors, reduce this value.
* `--trust_remote_code`: Required to load the MiniMax-M3 configuration and custom code from the checkpoint.
* `--reasoning_parser minimax_m3`: Parses the MiniMax-M3 reasoning trace, which is wrapped in `<mm:think>...</mm:think>` tags, into the structured `reasoning_content` field of chat responses.
* `--tool_parser minimax_m3`: Parses MiniMax-M3 tool/function calls from the model output.
* `--chat_template`: Path to the chat template shipped with the checkpoint (`chat_template.jinja`).
* `--extra_llm_api_options`: Path to the YAML file with additional LLM API options described below.

#### `--extra_llm_api_options` YAML options

These options provide control over TensorRT LLM's behavior and are set within the YAML file passed via the `--extra_llm_api_options` argument.

##### `sparse_attention_config`

* **Description**: Selects and configures the block-sparse attention backend.
* **Options**:
  * `algorithm`: Must be set to `minimax_m3` to run MiniMax-M3.

##### `kv_cache_config`

* **Description**: Configuration for the Key-Value (KV) cache.
* **Options**:
  * `enable_block_reuse`: Enables KV cache block reuse across requests with shared prefixes. **Must be `false`** for MiniMax-M3, as the sparse-attention path does not support KV cache reuse.

See the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full list of options which can be used in the YAML configuration file.

## Testing API Endpoint

The server (the OpenAI-compatible REST endpoint) runs on the rank-0 node, listening on port `8000`. Send requests to that node's hostname or IP; `localhost` only works from the rank-0 node itself. The examples below use `localhost:8000` — replace it with the rank-0 node address when querying from elsewhere.

### Health Check

You can query the health/readiness of the server using:

```bash
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When `Status: 200` is returned, the server is ready for queries. Note that the very first query may take longer due to initialization and compilation.

### Basic Test

After the TensorRT LLM server is set up and shows *Application startup complete*, you can send requests to the server.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "MiniMaxAI/MiniMax-M3",
      "prompt": "What is the capital of France?",
      "max_tokens": 16,
      "temperature": 0
  }'
```

Example response:

```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "model": "MiniMaxAI/MiniMax-M3",
  "choices": [
    {
      "index": 0,
      "text": "The capital of France is Paris. Paris is the largest city",
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 24,
    "completion_tokens": 16
  }
}
```

### Chat Completions Test

MiniMax-M3 is a reasoning model. Using the `/v1/chat/completions` endpoint with the configured chat template and reasoning parser, the reasoning trace is returned separately in the `reasoning_content` field.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "MiniMaxAI/MiniMax-M3",
      "messages": [
          {"role": "user", "content": "What is the capital of France?"}
      ],
      "max_tokens": 256,
      "temperature": 0
  }'
```

The `message` object in the response contains both the visible `content` and, when the model emits a `<mm:think>...</mm:think>` block, the parsed `reasoning_content`.

## Troubleshooting Tips

* **CUDA OOM errors:** Try reducing `--max_batch_size` or `--free_gpu_memory_fraction`.
  * As a workaround for memory fragmentation, you can set `PYTORCH_ALLOC_CONF=max_split_size_mb:8192`. For more details, refer to the [PyTorch documentation on optimizing memory usage](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf).
* **Sparse attention not applied:** Ensure `sparse_attention_config.algorithm` is set to `minimax_m3` in the YAML file passed to `--extra_llm_api_options`.
* **KV cache reuse errors:** Confirm `kv_cache_config.enable_block_reuse` is `false`; KV cache reuse is not supported on the sparse-attention path.
* **Model fails to load:** Make sure `--trust_remote_code` is set and that the checkpoint path is correct and fully downloaded (Git LFS).
* **Multi-node startup hangs or ranks can't find each other:** Verify that the model weights, `/workspace/extra_llm_api.yaml`, and all mounted paths resolve identically on both nodes (shared filesystem), that the total Slurm task count equals `--tp_size` (`8`), and that the inter-node interconnect (e.g., InfiniBand) is healthy.
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
        --model MiniMaxAI/MiniMax-M3 \
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
