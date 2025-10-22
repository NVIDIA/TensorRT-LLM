# Quick Start Recipe for GPT-OSS on TensorRT-LLM - Blackwell Hardware

## Introduction

This deployment guide provides step-by-step instructions for running the GPT-OSS model using TensorRT-LLM, optimized for NVIDIA GPUs. It covers the complete setup required; from accessing model weights and preparing the software environment to configuring TensorRT-LLM parameters, launching the server, and validating inference output.

The guide is intended for developers and practitioners seeking high-throughput or low-latency inference using NVIDIA’s accelerated stack—starting with the PyTorch container from NGC, then installing TensorRT-LLM for model serving.

## Prerequisites

* GPU: NVIDIA Blackwell Architecture
* OS: Linux
* Drivers: CUDA Driver 575 or Later
* Docker with NVIDIA Container Toolkit installed
* Python3 and python3-pip (Optional, for accuracy evaluation only)

## Models

* MXFP4 model: [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b)


## MoE Backend Support Matrix

There are multiple MOE backends inside TensorRT LLM. Here are the support matrix of the MOE backends.

| Device     | Activation Type | MoE Weights Type | MoE Backend | Use Case       |
|------------|------------------|------------------|-------------|----------------|
| B200/GB200 | MXFP8            | MXFP4            | TRTLLM      | Low Latency    |
| B200/GB200 | MXFP8            | MXFP4            | CUTLASS     | Max Throughput |

The default moe backend is `CUTLASS`, so for the combination which is not supported by `CUTLASS`, one must set the `moe_config.backend` explicitly to run the model.

## Deployment Steps

### Run Docker Container

Run the docker container using the TensorRT-LLM NVIDIA NGC image.

```shell
docker run --rm -it \
--ipc=host \
--gpus all \
-p 8000:8000 \
-v ~/.cache:/root/.cache:rw \
--name tensorrt_llm \
nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc6 \
/bin/bash
```

Note:

* The command mounts your user `.cache` directory to save the downloaded model checkpoints which are saved to `~/.cache/huggingface/hub/` by default. This prevents having to redownload the weights each time you rerun the container. If the `~/.cache` directory doesn’t exist please create it using `$ mkdir ~/.cache`.
* You can mount additional directories and paths using the `-v <host_path>:<container_path>` flag if needed, such as mounting the downloaded weight paths.
* The command also maps port `8000` from the container to your host so you can access the LLM API endpoint from your host
* See the <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags> for all the available containers. The containers published in the main branch weekly have `rcN` suffix, while the monthly release with QA tests has no `rcN` suffix. Use the `rc` release to get the latest model and feature support.

If you want to use latest main branch, you can choose to build from source to install TensorRT-LLM, the steps refer to <https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source-linux.html>.

### Creating the TensorRT LLM Server config

We create a YAML configuration file `/tmp/config.yml` for the TensorRT-LLM Server and populate it with the following recommended performance settings.

For low-latency with `TRTLLM` MOE backend:

```shell
EXTRA_LLM_API_FILE=/tmp/config.yml

cat << EOF > ${EXTRA_LLM_API_FILE}
enable_attention_dp: false
cuda_graph_config:
  enable_padding: true
  max_batch_size: 720
moe_config:
    backend: TRTLLM
stream_interval: 20
num_postprocess_workers: 4
EOF
```

For max-throughput with `CUTLASS` MOE backend:

```shell
EXTRA_LLM_API_FILE=/tmp/config.yml

cat << EOF > ${EXTRA_LLM_API_FILE}
enable_attention_dp: true
cuda_graph_config:
  enable_padding: true
  max_batch_size: 720
moe_config:
    backend: CUTLASS
stream_interval: 20
num_postprocess_workers: 4
attention_dp_config:
    enable_balance: true
    batching_wait_iters: 50
    timeout_iters: 1
EOF
```

### Launch the TensorRT LLM Server

Below is an example command to launch the TensorRT LLM server with the GPT-OSS model from within the container. The command is specifically configured for the 1024/1024 Input/Output Sequence Length test. The explanation of each flag is shown in the “Configs and Parameters” section.

```shell
trtllm-serve openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --max_batch_size 720 \
    --max_num_tokens 16384 \
    --kv_cache_free_gpu_memory_fraction 0.9 \
    --tp_size 8 \
    --ep_size 8 \
    --trust_remote_code \
    --extra_llm_api_options ${EXTRA_LLM_API_FILE}
```

After the server is set up, the client can now send prompt requests to the server and receive results.

### Configs and Parameters

These options are used directly on the command line when you start the `trtllm-serve` process.

#### `--tp_size`

* **Description:** Sets the **tensor-parallel size**. This should typically match the number of GPUs you intend to use for a single model instance.

#### `--ep_size`

* **Description:** Sets the **expert-parallel size** for Mixture-of-Experts (MoE) models. Like `tp_size`, this should generally match the number of GPUs you're using. This setting has no effect on non-MoE models.

#### `--kv_cache_free_gpu_memory_fraction`

* **Description:** A value between `0.0` and `1.0` that specifies the fraction of free GPU memory to reserve for the KV cache after the model is loaded. Since memory usage can fluctuate, this buffer helps prevent out-of-memory (OOM) errors.
* **Recommendation:** If you experience OOM errors, try reducing this value to `0.7` or lower.

#### `--max_batch_size`

* **Description:** The maximum number of user requests that can be grouped into a single batch for processing. The actual max batch size that can be achieved depends on total sequence length (input + output).

#### `--max_num_tokens`

* **Description:** The maximum total number of tokens (across all requests) allowed inside a single scheduled batch.

#### `--max_seq_len`

* **Description:** The maximum possible sequence length for a single request, including both input and generated output tokens. We won't specifically set it. It will be inferred from model config.

#### `--trust_remote_code`

* **Description:** Allows TensorRT-LLM to download models and tokenizers from Hugging Face. This flag is passed directly to the Hugging Face API.


#### Extra LLM API Options (YAML Configuration)

These options provide finer control over performance and are set within a YAML file passed to the `trtllm-serve` command via the `--extra_llm_api_options` argument.

#### `cuda_graph_config`

* **Description**: A section for configuring CUDA graphs to optimize performance.

* **Options**:

  * `enable_padding`: If `"true"`, input batches are padded to the nearest `cuda_graph_batch_size`. This can significantly improve performance.

    **Default**: `false`

  * `max_batch_size`: Sets the maximum batch size for which a CUDA graph will be created.

    **Default**: `0`

    **Recommendation**: Set this to the same value as the `--max_batch_size` command-line option.

#### `moe_config`

* **Description**: Configuration for Mixture-of-Experts (MoE) models.

* **Options**:

  * `backend`: The backend to use for MoE operations.
    **Default**: `CUTLASS`

See the [`TorchLlmArgs` class](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) for the full list of options which can be used in the `extra_llm_api_options`.

## Testing API Endpoint

### Basic Test

Start a new terminal on the host to test the TensorRT-LLM server you just launched.

You can query the health/readiness of the server using:

```shell
curl -s -o /dev/null -w "Status: %{http_code}\n" "http://localhost:8000/health"
```

When the `Status: 200` code is returned, the server is ready for queries. Note that the very first query may take longer due to initialization and compilation.

After the TensorRT LLM server is set up and shows Application startup complete, you can send requests to the server.

```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json"  -d '{
    "model": "openai/gpt-oss-120b",
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

Here is an example response, showing that the TensorRT LLM server reasons and answers the questions.

TODO: Use Chat Compeletions API / Responses API as the example after the PR is merged.

```json
{"id":"chatcmpl-c5bf51b5cab94e10ba5da5266d12ee59","object":"chat.completion","created":1755815898,"model":"openai/gpt-oss-120b","choices":[{"index":0,"message":{"role":"assistant","content":"analysisThe user asks: \"Where is New York?\" Likely they want location info. Provide answer: New York State in northeastern US, New York City on the east coast, coordinates, etc. Provide context.assistantfinal**New York** can refer to two related places in the United States:\n\n| What it is | Where it is | Approx. coordinates | How to picture it |\n|------------|------------|--------------------|-------------------|\n| **New York State** | The northeastern corner of the United States, bordered by **Vermont, Massachusetts, Connecticut, New Jersey, Pennsylvania, and the Canadian provinces of Ontario and Quebec**. | 42.7° N, 75.5° W (roughly the state’s geographic centre) | A roughly rectangular state that stretches from the Atlantic Ocean in the southeast to the Adirondack Mountains and the Great Lakes region in the north. |\n| **New York City (NYC)** | The largest city in the state, located on the **southern tip of the state** where the **Hudson River meets the Atlantic Ocean**. It occupies five boroughs: Manhattan, Brooklyn, Queens, The Bronx, and Staten Island. | 40.7128° N, 74.0060° W | A dense, world‑famous metropolis that sits on a series of islands (Manhattan, Staten Island, parts of the Bronx) and the mainland (Brooklyn and Queens). |\n\n### Quick geographic context\n- **On a map of the United States:** New York State is in the **Northeast** region, just east of the Great Lakes and north of Pennsylvania.  \n- **From Washington, D.C.:** Travel roughly **225 mi (360 km) northeast**.  \n- **From Boston, MA:** Travel about **215 mi (350 km) southwest**.  \n- **From Toronto, Canada:** Travel about **500 mi (800 km) southeast**.\n\n### Travel tips\n- **By air:** Major airports include **John F. Kennedy International (JFK)**, **LaGuardia (LGA)**, and **Newark Liberty International (EWR)** (the latter is actually in New Jersey but serves the NYC metro area).  \n- **By train:** Amtrak’s **Northeast Corridor** runs from **Boston → New York City → Washington, D.C.**  \n- **By car:** Interstates **I‑87** (north‑south) and **I‑90** (east‑west) are the primary highways crossing the state.\n\n### Fun fact\n- The name “**New York**” was given by the English in 1664, honoring the Duke of York (later King James II). The city’s original Dutch name was **“New Amsterdam.”**\n\nIf you need more specific directions (e.g., how to get to a particular neighborhood, landmark, or the state capital **Albany**), just let me know!","reasoning_content":null,"tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null,"mm_embedding_handle":null,"disaggregated_params":null,"avg_decoded_tokens_per_iter":1.0}],"usage":{"prompt_tokens":72,"total_tokens":705,"completion_tokens":633},"prompt_token_ids":null}
```

### Troubleshooting Tips

* If you encounter CUDA out-of-memory errors, try reducing `max_batch_size` or `max_seq_len`.
* Ensure your model checkpoints are compatible with the expected format.
* For performance issues, check GPU utilization with nvidia-smi while the server is running.
* If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed.
* For connection issues, make sure the server port (`8000` in this guide) is not being used by another application.

### Running Evaluations to Verify Accuracy (Optional)

We use OpenAI's official evaluation tool to test the model's accuracy. For more information see [https://github.com/openai/gpt-oss/tree/main/gpt_oss/evals](gpt-oss-eval).
With the added support of Chat Completions and Responses API in `trtllm-serve,` `gpt_oss.evals` works directly without any modifications.

You need to set `enable_attention_dp`, `tp_size`, `ep_size`, `max_batch_size` and `max_num_tokens` when launching the trtllm server and set `reasoning-effort` when launching evaluation in gpt-oss. Below are some reference configurations for accuracy evaluation on B200.

| **reasoning-effort** | **parallel configuration** | **max_batch_size** | **max_num_tokens** |
|:--------------------:|:--------------------------:|:------------------:|:------------------:|
| low/medium           | DEP8 / DEP4                | 128                | 32768              |
| high                 | DEP8 / DEP4                | 2                  | 133120             |
| low/medium           | TP8 / TP4                  | 1024               | 32768              |
| high                 | TP8 / TP4                  | 720                | 133120             |

Below is an example command for evaluating the accuracy of gpt-oss-120b with low and medium reasoning-effort on GPQA and AIME2025.

```shell
# execute this command in gpt-oss
python -m gpt_oss.evals \
  --sampler chat_completions \
  --eval gpqa,aime25 \
  --model gpt-oss-120b \
  --reasoning-effort low,medium
```



## Benchmarking Performance

To benchmark the performance of your TensorRT-LLM server you can leverage the built-in `benchmark_serving.py` script. To do this first creating a wrapper `bench.sh` script.

```shell
cat <<'EOF' > bench.sh
#!/usr/bin/env bash
set -euo pipefail

concurrency_list="32 64 128 256 512 1024 2048 4096"
multi_round=5
isl=1024
osl=1024
result_dir=/tmp/gpt_oss_output

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model openai/gpt-oss-120b \
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

To achieve max through-put, with attention DP on, one needs to sweep up to `concurrency = max_batch_size * num_gpus`.

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

Sample TensorRT-LLM serving benchmark output. Your results may vary due to ongoing software optimizations.

```
============ Serving Benchmark Result ============
Successful requests:                      16
Benchmark duration (s):                   17.66
Total input tokens:                       16384
Total generated tokens:                   16384
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
  * The typical time elapsed from when a request is sent until the first output token is generated.

#### Time Per Output Token (TPOT) and Inter-Token Latency (ITL)
  * TPOT is the typical time required to generate each token *after* the first one.
  * ITL is the typical time delay between the completion of one token and the completion of the next.
  * Both TPOT and ITL ignore TTFT.

For a single request, ITLs are the time intervals between tokens, while TPOT is the average of those intervals:

```math
\text{TPOT (1\ request)} = \text{Avg(ITL)} = \frac{\text{E2E\ latency} - \text{TTFT}}{\text{\#Output\ Tokens} - 1}
```

Across different requests, **average TPOT** is the mean of each request's TPOT (all requests weighted equally), while **average ITL** is token-weighted (all tokens weighted equally):

```math
\text{Avg TPOT (N requests)} = \frac{\text{TPOT}_1 + \text{TPOT}_2 + \cdots + \text{TPOT}_N}{N}
```

```math
\text{Avg ITL (N requests)} = \frac{\text{Sum of all ITLs across requests}}{\text{\#Output Tokens across requests}}
```

#### End-to-End (E2E) Latency
  * The typical total time from when a request is submitted until the final token of the response is received.

#### Total Token Throughput
  * The combined rate at which the system processes both input (prompt) tokens and output (generated) tokens.
```math
\text{Total\ TPS} = \frac{\text{\#Input\ Tokens}+\text{\#Output\ Tokens}}{T_{last} - T_{first}}
```

#### Tokens Per Second (TPS) or Output Token Throughput
  * how many output tokens the system generates each second.
```math
\text{TPS} = \frac{\text{\#Output\ Tokens}}{T_{last} - T_{first}}
```
