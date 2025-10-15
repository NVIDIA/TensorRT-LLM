# Quick Start Recipe for Llama4 Scout 17B on TensorRT LLM - Blackwell & Hopper Hardware

## Introduction

This deployment guide provides step-by-step instructions for running the Llama-4-Scout-17B-16E-Instruct model using TensorRT LLM with FP8 and NVFP4 quantization, optimized for NVIDIA GPUs. It covers the complete setup required; from accessing model weights and preparing the software environment to configuring TensorRT LLM parameters, launching the server, and validating inference output.

The guide is intended for developers and practitioners seeking high-throughput or low-latency inference using NVIDIA’s accelerated stack—starting with the PyTorch container from NGC, then installing TensorRT LLM for model serving, FlashInfer for optimized CUDA kernels, and ModelOpt to enable FP8 and NVFP4 quantized execution.

## Access & Licensing

To use Llama4 Scout 17B, you must first agree to Meta’s [Llama 4 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE). NVIDIA’s quantized versions (FP8 and NVFP4) are built on top of the base model and are available for research and commercial use under the same license.

## Prerequisites

* GPU: NVIDIA Blackwell or Hopper Architecture
* OS: Linux
* Drivers: CUDA Driver 575 or Later
* Docker with NVIDIA Container Toolkit installed
* Python3 and python3-pip (Optional, for accuracy evaluation only)

## Models

* FP8 model: [Llama-4-Scout-17B-16E-Instruct-FP8](https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-FP8)
* NVFP4 model: [Llama-4-Scout-17B-16E-Instruct-FP4](https://huggingface.co/nvidia/Llama-4-Scout-17B-16E-Instruct-FP4)

Note that NVFP4 is only supported on NVIDIA Blackwell platform.

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
nvcr.io/nvidia/tensorrt-llm/release:1.0.0rc6 \
/bin/bash
```

Note:

* The command mounts your user `.cache` directory to save the downloaded model checkpoints which are saved to `~/.cache/huggingface/hub/` by default. This prevents having to redownload the weights each time you rerun the container. If the `~/.cache` directory doesn’t exist please create it using `$ mkdir ~/.cache`.
* You can mount additional directories and paths using the `-v <host_path>:<container_path>` flag if needed, such as mounting the downloaded weight paths.
* The command also maps port `8000` from the container to your host so you can access the LLM API endpoint from your host
* See the <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release/tags> for all the available containers. The containers published in the main branch weekly have `rcN` suffix, while the monthly release with QA tests has no `rcN` suffix. Use the `rc` release to get the latest model and feature support.

If you want to use latest main branch, you can choose to build from source to install TensorRT LLM, the steps refer to [https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source-linux.html](https://nvidia.github.io/TensorRT-LLM/latest/installation/build-from-source-linux.html)

### Creating the TensorRT LLM Server config

We create a YAML configuration file /tmp/config.yml for the TensorRT LLM Server and populate it with the following recommended performance settings.

```shell
EXTRA_LLM_API_FILE=/tmp/config.yml

cat << EOF > ${EXTRA_LLM_API_FILE}
enable_attention_dp: false
cuda_graph_config:
  enable_padding: true
  max_batch_size: 1024
kv_cache_config:
  dtype: fp8
EOF
```

### Launch the TensorRT LLM Server

Below is an example command to launch the TensorRT LLM server with the Llama-4-Scout-17B-16E-Instruct-FP8 model from within the container. The command is specifically configured for the 1024/1024 Input/Output Sequence Length test. The explanation of each flag is shown in the “Configs and Parameters” section.

```shell
trtllm-serve nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 \
    --host 0.0.0.0 \
    --port 8000 \
    --max_batch_size 1024 \
    --max_num_tokens 2048 \
    --max_seq_len 2048 \
    --kv_cache_free_gpu_memory_fraction 0.9 \
    --tp_size 1 \
    --ep_size 1 \
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

* **Description:** The maximum number of user requests that can be grouped into a single batch for processing.

#### `--max_num_tokens`

* **Description:** The maximum total number of tokens (across all requests) allowed inside a single scheduled batch.

#### `--max_seq_len`

* **Description:** The maximum possible sequence length for a single request, including both input and generated output tokens.

#### `--trust_remote_code`

&emsp;**Description:** Allows TensorRT LLM to download models and tokenizers from Hugging Face. This flag is passed directly to the Hugging Face API.


#### Extra LLM API Options (YAML Configuration)

These options provide finer control over performance and are set within a YAML file passed to the `trtllm-serve` command via the `--extra_llm_api_options` argument.

#### `kv_cache_config`

* **Description**: A section for configuring the Key-Value (KV) cache.

* **Options**:

  * `dtype`: Sets the data type for the KV cache.
    **Default**: `"auto"` (uses the data type specified in the model checkpoint).

#### `cuda_graph_config`

* **Description**: A section for configuring CUDA graphs to optimize performance.

* **Options**:

  * `enable_padding`: If `"true"`, input batches are padded to the nearest `cuda_graph_batch_size`. This can significantly improve performance.

    **Default**: `false`

  * `max_batch_size`: Sets the maximum batch size for which a CUDA graph will be created.

    **Default**: `0`

    **Recommendation**: Set this to the same value as the `--max_batch_size` command-line option.

  * `batch_sizes`: A specific list of batch sizes to create CUDA graphs for.

     **Default**: `None`

#### `moe_config`

* **Description**: Configuration for Mixture-of-Experts (MoE) models.

* **Options**:

  * `backend`: The backend to use for MoE operations.
    **Default**: `CUTLASS`

#### `attention_backend`

* **Description**: The backend to use for attention calculations.

* **Default**: `TRTLLM`

See the [TorchLlmArgs](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html#tensorrt_llm.llmapi.TorchLlmArgs) class for the full list of options which can be used in the `extra_llm_api_options`.

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
curl http://localhost:8000/v1/completions -H "Content-Type: application/json"  -d '{
      "model": "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
      "prompt": "Where is New York?",
      "max_tokens": 16,
      "temperature": 0
}'
```

Here is an example response, showing that the TensorRT LLM server returns “New York is a state located in the northeastern United States. It is bordered by”, completing the input sequence.

```json
{"id":"cmpl-bc1393d529ce485c961d9ffee5b25d72","object":"text_completion","created":1753843963,"model":"$MODEL","choices":[{"index":0,"text":" New York is a state located in the northeastern United States. It is bordered by","token_ids":null,"logprobs":null,"context_logits":null,"finish_reason":"length","stop_reason":null,"disaggregated_params":null}],"usage":{"prompt_tokens":6,"total_tokens":22,"completion_tokens":16},"prompt_token_ids":null}
```

### Troubleshooting Tips

* If you encounter CUDA out-of-memory errors, try reducing `max_batch_size` or `max_seq_len`.
* Ensure your model checkpoints are compatible with the expected format.
* For performance issues, check GPU utilization with nvidia-smi while the server is running.
* If the container fails to start, verify that the NVIDIA Container Toolkit is properly installed.
* For connection issues, make sure the server port (`8000` in this guide) is not being used by another application.

### Running Evaluations to Verify Accuracy (Optional)

We use the lm-eval tool to test the model’s accuracy. For more information see <https://github.com/EleutherAI/lm-evaluation-harness>.

To run the evaluation harness exec into the running TensorRT LLM container and install with this command:

```shell
docker exec -it tensorrt_llm /bin/bash

pip install -U lm-eval
```

FP8 command for GSM8K

```shell
MODEL_PATH=nvidia/Llama-4-Scout-17B-16E-Instruct-FP8

lm_eval --model local-completions  --tasks gsm8k --batch_size 256 --gen_kwargs temperature=0.0 --num_fewshot 5 --model_args model=${MODEL_PATH},base_url=http://localhost:8000/v1/completions,num_concurrent=32,max_retries=20,tokenized_requests=False --log_samples --output_path trtllm.fp8.gsm8k
```

Sample result in Blackwell.

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9189|±  |0.0075|
|     |       |strict-match    |     5|exact_match|↑  |0.8984|±  |0.0083|
```

FP4 command for GSM8K

```shell
MODEL_PATH=nvidia/Llama-4-Scout-17B-16E-Instruct-FP4

lm_eval --model local-completions  --tasks gsm8k --batch_size 256 --gen_kwargs temperature=0.0 --num_fewshot 5 --model_args model=${MODEL_PATH},base_url=http://localhost:8000/v1/completions,num_concurrent=32,max_retries=20,tokenized_requests=False --log_samples --output_path trtllm.fp4.gsm8k
```

Sample result in Blackwell

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9075|±  |0.0080|
|     |       |strict-match    |     5|exact_match|↑  |0.8908|±  |0.0086|
```

## Benchmarking Performance

To benchmark the performance of your TensorRT LLM server you can leverage the built-in `benchmark_serving.py` script. To do this first creating a wrapper `bench.sh`(http://bench.sh) script.

```shell
cat <<EOF >  bench.sh
concurrency_list="1 2 4 8 16 32 64 128 256"
multi_round=5
isl=1024
osl=1024
result_dir=/tmp/llama4_output

for concurrency in ${concurrency_list}; do
    num_prompts=$((concurrency * multi_round))
    python -m tensorrt_llm.serve.scripts.benchmark_serving \
        --model nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 \
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

To benchmark the FP4 model, replace `--model nvidia/Llama-4-Scout-17B-16E-Instruct-FP8` with `--model nvidia/Llama-4-Scout-17B-16E-Instruct-FP4`.

If you want to save the results to a file add the following options.

```shell
--save-result \
--result-dir "${result_dir}" \
--result-filename "concurrency_${concurrency}.json"
```

For more benchmarking options see [benchmark_serving.py](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/benchmark_serving.py)

Run bench.sh to begin a serving benchmark. This will take a long time if you run all the concurrencies mentioned in the above bench.sh script.

```shell
./bench.sh
```

Sample TensorRT LLM serving benchmark output. Your results may vary due to ongoing software optimizations.

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
